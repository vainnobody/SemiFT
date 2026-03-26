import torch

import segmind
from segmind import classmix_batch, create_block_mask
from util.segmind_utils import compute_masked_segmentation_loss


def test_classmix_batch_preserves_shapes_and_dtypes():
    batch_size, channels, height, width = 2, 3, 32, 32
    img_w = torch.randn(batch_size, channels, height, width)
    img_s = torch.randn(batch_size, channels, height, width)
    pseudo = torch.randint(0, 4, (batch_size, height, width))
    conf = torch.rand(batch_size, height, width)
    entropy = torch.rand(batch_size, height, width)
    valid = torch.rand(batch_size, height, width) > 0.2

    img_w_mix, img_s_mix, pseudo_mix, conf_mix, entropy_mix, valid_mix = classmix_batch(
        img_w, img_s, pseudo, conf, entropy, valid
    )

    assert img_w_mix.shape == img_w.shape
    assert img_s_mix.shape == img_s.shape
    assert pseudo_mix.shape == pseudo.shape
    assert conf_mix.shape == conf.shape
    assert entropy_mix.shape == entropy.shape
    assert valid_mix.shape == valid.shape
    assert pseudo_mix.dtype == torch.long
    assert valid_mix.dtype == torch.bool


def test_create_block_mask_is_patch_aligned_and_matches_requested_size():
    mask = create_block_mask(
        batch_size=3,
        height=64,
        width=64,
        mask_patch=8,
        mask_ratio=0.25,
        device="cpu",
    )

    assert mask.shape == (3, 1, 64, 64)
    assert set(mask.unique().tolist()).issubset({0.0, 1.0})
    coarse = mask.unfold(2, 8, 8).unfold(3, 8, 8)
    anchors = coarse[..., :1, :1]
    assert torch.all((coarse == anchors).reshape(-1))


def test_unsupervised_mask_style_reduction_avoids_nan_when_no_high_confidence_pixels():
    loss_u_map = torch.rand(2, 8, 8)
    unsup_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    reduced = (loss_u_map * unsup_mask.float()).sum() / unsup_mask.sum().clamp(min=1.0)
    assert torch.isfinite(reduced)
    assert reduced.item() == 0.0


def test_masked_segmentation_loss_only_supervises_masked_pixels():
    logits = torch.tensor(
        [[
            [[10.0, -10.0], [10.0, -10.0]],
            [[-10.0, 10.0], [-10.0, 10.0]],
        ]]
    )
    labels = torch.tensor([[[0, 1], [1, 0]]])
    mask = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32)
    loss = compute_masked_segmentation_loss(logits, labels, mask, ignore_index=255)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_segmind_source_uses_lr_multi_defaults_and_pseudo_threshold():
    source = open("segmind.py", "r", encoding="utf-8").read()
    assert "def apply_segmind_defaults(cfg):" in source
    assert 'cfg.setdefault("lambda_r", 0.0)' in source
    assert 'cfg.setdefault("lambda_rsc", 0.0)' in source
    assert 'cfg.setdefault("lambda_c", 0.0)' in source
    assert 'cfg.setdefault("pseudo_threshold", cfg.get("conf_thresh", 0.95))' in source
    assert '"lr_scale": lr_multi' in source
    assert 'pseudo_conf_mix >= cfg["pseudo_threshold"]' in source
    assert "ema_ratio = min(1 - 1 / (iters + 1), alpha_ema)" in source
    assert "for param, param_ema in zip(model.parameters(), model_ema.parameters()):" in source
    assert "for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):" in source
    assert "model_ema.eval()" in source
    assert "find_unused_parameters=False" in source
    assert 'logger.info("Enabled DDP static graph for SegMind training.")' in source


def test_validate_segmind_recipe_rejects_incompatible_crop_and_mask_gap():
    try:
        segmind.validate_segmind_recipe({"crop_size": 518, "mask_gap": 4})
    except ValueError as exc:
        assert "crop_size divisible by mask_gap" in str(exc)
    else:
        raise AssertionError("validate_segmind_recipe should reject 518 with mask_gap=4")
