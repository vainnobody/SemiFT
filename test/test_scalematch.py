import sys
import types
from pathlib import Path

import pytest
import torch


stub_tb = types.ModuleType("torch.utils.tensorboard")


class StubSummaryWriter:
    def __init__(self, *args, **kwargs):
        self.scalars = []

    def add_scalar(self, *args, **kwargs):
        self.scalars.append((args, kwargs))


stub_tb.SummaryWriter = StubSummaryWriter
sys.modules.setdefault("torch.utils.tensorboard", stub_tb)

stub_supervised = types.ModuleType("supervised")


def _validation_cpu(*args, **kwargs):
    return 0.0, []


stub_supervised.validation_cpu = _validation_cpu
sys.modules.setdefault("supervised", stub_supervised)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scalematch
from model.semseg import dpt_scalematch
from model.semseg import upernet_scalematch


class FakeBackbone(torch.nn.Module):
    def __init__(self, model_name="small"):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = 8
        self.patch_size = 14

    def get_intermediate_layers(self, x, idxs):
        batch_size = x.shape[0]
        patch_h = x.shape[-2] // self.patch_size
        patch_w = x.shape[-1] // self.patch_size
        num_tokens = patch_h * patch_w
        base = torch.linspace(
            0.0,
            1.0,
            batch_size * num_tokens * self.embed_dim,
            device=x.device,
            dtype=x.dtype,
        ).reshape(batch_size, num_tokens, self.embed_dim)
        return tuple(base + float(i) for i, _ in enumerate(idxs))


@pytest.fixture(autouse=True)
def patch_fake_backbones(monkeypatch):
    monkeypatch.setattr(dpt_scalematch, "DINOv2", FakeBackbone)
    monkeypatch.setattr(dpt_scalematch, "DINOv3", FakeBackbone)
    monkeypatch.setattr(upernet_scalematch, "DINOv2", FakeBackbone)
    monkeypatch.setattr(upernet_scalematch, "DINOv3", FakeBackbone)


def build_test_model():
    return dpt_scalematch.DPT_ScaleMatch(
        encoder_size="small",
        nclass=3,
        features=16,
        out_channels=[8, 8, 8, 8],
        backbone_version="dinov2",
    )


def build_test_upernet_model():
    return upernet_scalematch.UperNet_ScaleMatch(
        encoder_size="small",
        nclass=3,
        fpn_channels=16,
        backbone_version="dinov2",
    )


def test_scalematch_recipe_uses_cityscapes_defaults():
    recipe = scalematch.get_scalematch_recipe({"dataset": "cityscapes"})
    assert recipe["conf_thresh"] == 0.0
    assert recipe["img_scales"] == [0.5, 0.75, 1.0, 1.25]
    assert recipe["feat_l_scales"] == [1.0, 1.25, 1.5]
    assert recipe["warm_up"] == 10


def test_scalematch_remote_dataset_repeat_factor_len():
    dataset = scalematch.ScaleMatchRemoteSemiDataset.__new__(
        scalematch.ScaleMatchRemoteSemiDataset
    )
    dataset.ids = ["a", "b", "c"]
    dataset.epoch_repeat_factor = 4
    assert len(dataset) == 12


def test_build_scalematch_model_rejects_non_dpt():
    cfg = {"backbone": "dinov2_small", "nclass": 3, "model": "upernet"}
    model, backbone_version = scalematch.build_scalematch_model(cfg)
    assert isinstance(model, upernet_scalematch.UperNet_ScaleMatch)
    assert backbone_version == "dinov2"


def test_build_scalematch_model_rejects_unknown_model():
    cfg = {"backbone": "dinov2_small", "nclass": 3, "model": "unknown"}
    with pytest.raises(ValueError, match="supports only 'dpt' and 'upernet'"):
        scalematch.build_scalematch_model(cfg)


def test_dpt_scalematch_forward_outputs_expected_shapes():
    model = build_test_model()
    model.eval()
    x = torch.randn(2, 3, 56, 56)

    logits = model(x, scale_factor=None, feature_scale=1.25)
    assert logits.shape == (2, 3, 56, 56)

    hi = model(x, scale_factor=1.5, feature_scale=0.75)
    lo = model(x, scale_factor=0.5, feature_scale=1.25)
    for out in (hi, lo):
        assert set(out) == {"pred_joint", "pred_ori", "pred_fp", "pred_size"}
        for value in out.values():
            assert value.shape == (2, 3, 56, 56)


def test_dpt_scalematch_training_outputs_include_pseudo_logits():
    model = build_test_model()
    model.train()
    x = torch.randn(4, 3, 56, 56)
    strong = torch.randn(2, 3, 56, 56)

    out_ori = model(
        x,
        scale_factor=1.5,
        feature_scale=0.75,
        strong_inputs=strong,
        pseudo_mode="ori",
    )
    out_joint = model(
        x,
        scale_factor=0.5,
        feature_scale=1.25,
        strong_inputs=strong,
        pseudo_mode="joint",
    )

    expected_keys = {
        "pred_joint",
        "pred_ori",
        "pred_size",
        "pred_fp",
        "pred_strong",
        "pseudo_logits",
    }
    assert set(out_ori) == expected_keys
    assert set(out_joint) == expected_keys
    assert out_ori["pred_strong"].shape == (2, 3, 56, 56)
    assert out_joint["pred_strong"].shape == (2, 3, 56, 56)
    assert torch.equal(out_ori["pseudo_logits"], out_ori["pred_ori"].detach())
    assert torch.equal(out_joint["pseudo_logits"], out_joint["pred_joint"].detach())


def test_upernet_scalematch_forward_outputs_expected_shapes():
    model = build_test_upernet_model()
    model.eval()
    x = torch.randn(2, 3, 56, 56)

    logits = model(x, scale_factor=None, feature_scale=1.25)
    assert logits.shape == (2, 3, 56, 56)

    hi = model(x, scale_factor=1.5, feature_scale=0.75)
    lo = model(x, scale_factor=0.5, feature_scale=1.25)
    for out in (hi, lo):
        assert set(out) == {"pred_joint", "pred_ori", "pred_fp", "pred_size"}
        for value in out.values():
            assert value.shape == (2, 3, 56, 56)


def test_upernet_scalematch_training_outputs_include_pseudo_logits():
    model = build_test_upernet_model()
    model.train()
    x = torch.randn(4, 3, 56, 56)
    strong = torch.randn(2, 3, 56, 56)

    out_ori = model(
        x,
        scale_factor=1.5,
        feature_scale=0.75,
        strong_inputs=strong,
        pseudo_mode="ori",
    )
    out_joint = model(
        x,
        scale_factor=0.5,
        feature_scale=1.25,
        strong_inputs=strong,
        pseudo_mode="joint",
    )

    expected_keys = {
        "pred_joint",
        "pred_ori",
        "pred_size",
        "pred_fp",
        "pred_strong",
        "pseudo_logits",
    }
    assert set(out_ori) == expected_keys
    assert set(out_joint) == expected_keys
    assert out_ori["pred_strong"].shape == (2, 3, 56, 56)
    assert out_joint["pred_strong"].shape == (2, 3, 56, 56)
    assert torch.equal(out_ori["pseudo_logits"], out_ori["pred_ori"].detach())
    assert torch.equal(out_joint["pseudo_logits"], out_joint["pred_joint"].detach())


def test_compute_masked_class_hist_and_ratio():
    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    valid_mask = torch.tensor([[True, False, True], [False, True, True]])

    hist = scalematch.compute_masked_class_hist(labels, 3, valid_mask)
    ratio = scalematch.compute_class_ratio(labels, 3, valid_mask)

    assert torch.equal(hist, torch.tensor([2.0, 1.0, 1.0]))
    assert torch.allclose(ratio, torch.tensor([0.5, 0.25, 0.25]))


def test_masked_agreement_uses_valid_pixels_only():
    pred_a = torch.tensor([[0, 1], [2, 2]])
    pred_b = torch.tensor([[0, 0], [2, 1]])
    valid_mask = torch.tensor([[True, False], [True, True]])

    agreement = scalematch.masked_agreement(pred_a, pred_b, valid_mask)

    assert torch.isclose(agreement, torch.tensor(2.0 / 3.0))


def test_collect_debug_metrics_returns_expected_keys_and_shapes():
    nclass = 3
    num_lb = 1
    pred_u_w = torch.tensor(
        [
            [
                [[4.0, 0.0], [0.0, 4.0]],
                [[0.0, 4.0], [4.0, 0.0]],
                [[-1.0, -1.0], [-1.0, -1.0]],
            ]
        ]
    )
    student_out = {
        "pred_ori": torch.tensor(
            [
                [[[4.0, 1.0], [1.0, 4.0]], [[1.0, 4.0], [4.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [4.0, 0.0]], [[-1.0, -1.0], [-1.0, -1.0]]],
            ]
        ),
        "pred_joint": torch.tensor(
            [
                [[[1.0, 4.0], [4.0, 1.0]], [[4.0, 1.0], [1.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [4.0, 0.0]], [[-1.0, -1.0], [-1.0, -1.0]]],
            ]
        ),
    }
    pred_u_s = pred_u_w.clone()
    pred_x_joint = student_out["pred_joint"][:1]
    pred_x_ori = student_out["pred_ori"][:1]
    mask_u_w_cutmixed1 = torch.tensor([[[0, 1], [1, 0]]])
    conf_u_w = torch.tensor([[[0.99, 0.98], [0.97, 0.96]]])
    conf_u_w_cutmixed1 = conf_u_w.clone()
    valid_mask = torch.tensor([[[True, True], [True, False]]])
    ignore_mask_cutmixed1 = torch.tensor([[[0, 0], [0, 255]]])

    metrics = scalematch.collect_debug_metrics(
        pred_u_w=pred_u_w,
        student_out=student_out,
        pred_u_s=pred_u_s,
        pred_x_joint=pred_x_joint,
        pred_x_ori=pred_x_ori,
        mask_u_w_cutmixed1=mask_u_w_cutmixed1,
        conf_u_w=conf_u_w,
        conf_u_w_cutmixed1=conf_u_w_cutmixed1,
        valid_mask=valid_mask,
        ignore_mask_cutmixed1=ignore_mask_cutmixed1,
        ignore_index=255,
        conf_thresh=0.95,
        num_lb=num_lb,
        nclass=nclass,
    )

    expected_scalar_keys = {
        "teacher_vs_student_ori_agreement",
        "teacher_vs_student_joint_agreement",
        "student_joint_vs_ori_agreement",
        "strong_vs_pseudo_agreement",
        "conf_teacher_pseudo",
        "conf_student_ori_u",
        "conf_student_joint_u",
        "conf_student_strong",
    }
    expected_ratio_keys = {
        "pseudo_ratio",
        "accepted_pseudo_ratio",
        "student_joint_ratio",
        "student_ori_ratio",
        "strong_ratio",
        "labeled_joint_ratio",
        "labeled_ori_ratio",
    }

    assert expected_scalar_keys.issubset(metrics.keys())
    assert expected_ratio_keys.issubset(metrics.keys())
    for key in expected_ratio_keys:
        assert metrics[key].shape == (nclass,)
        assert torch.isclose(metrics[key].sum(), torch.tensor(1.0))


def test_compute_official_scalematch_total_loss_matches_official_weights():
    loss_x = torch.tensor(2.0)
    loss_u_s1 = torch.tensor(4.0)
    loss_u_size = torch.tensor(6.0)
    loss_u_w_fp = torch.tensor(8.0)

    total = scalematch.compute_official_scalematch_total_loss(
        loss_x, loss_u_s1, loss_u_size, loss_u_w_fp
    )

    expected = torch.tensor((2.0 + 0.25 * 4.0 + 0.25 * 6.0 + 0.5 * 8.0) / 2.0)
    assert torch.isclose(total, expected)


def test_select_pseudo_logits_from_student_out_uses_official_warmup_rule():
    student_out = {
        "pred_ori": torch.arange(24, dtype=torch.float32).reshape(4, 3, 2, 1),
        "pred_joint": torch.arange(24, 48, dtype=torch.float32).reshape(4, 3, 2, 1),
    }

    warmup_logits = scalematch.select_pseudo_logits_from_student_out(
        student_out, num_lb=1, epoch=0, warm_up=10
    )
    post_warmup_logits = scalematch.select_pseudo_logits_from_student_out(
        student_out, num_lb=1, epoch=10, warm_up=10
    )

    assert torch.equal(warmup_logits, student_out["pred_ori"][1:])
    assert torch.equal(post_warmup_logits, student_out["pred_joint"][1:])
    assert warmup_logits.requires_grad is False
    assert post_warmup_logits.requires_grad is False


def test_min_epoch_repeat_factor_for_nonempty_loader_matches_ddp_requirement():
    assert scalematch.min_epoch_repeat_factor_for_nonempty_loader(
        base_num_ids=1, world_size=8, batch_size=4
    ) == 32
    assert scalematch.min_epoch_repeat_factor_for_nonempty_loader(
        base_num_ids=12, world_size=8, batch_size=4
    ) == 3
    assert scalematch.min_epoch_repeat_factor_for_nonempty_loader(
        base_num_ids=17, world_size=8, batch_size=4
    ) == 2


def test_build_loader_guard_message_surfaces_actionable_fix():
    message = scalematch.build_loader_guard_message(
        dataset_name="vaihingen",
        split_name="train_u",
        base_num_ids=12,
        effective_num_ids=12,
        loader_len=0,
        world_size=8,
        batch_size=4,
        epoch_repeat_factor=1,
    )

    assert "loader has zero batches under DDP" in message
    assert "reducing --nproc_per_node" in message
    assert "increasing epoch_repeat_factor to at least 3" in message
