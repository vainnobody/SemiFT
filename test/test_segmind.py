import sys
import types
from pathlib import Path

import torch

stub_tb = types.ModuleType("torch.utils.tensorboard")


class StubSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass


stub_tb.SummaryWriter = StubSummaryWriter
sys.modules.setdefault("torch.utils.tensorboard", stub_tb)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.semseg import dpt as dpt_mod
from model.semseg import upernet as upernet_mod
from model.semseg.segmind import SegMindModel
from segmind import apply_ignore_mask_to_labels, build_entropy_targets
from util.segmind_utils import class_mix_batch, create_memory_bank, generate_grid_mask, segmind_contrastive_loss


class FakeResNet101Backbone(torch.nn.Module):
    feature_kind = "feature_map"
    out_channels = [256, 512, 1024, 2048]
    output_stride = 32
    patch_size = 32
    embed_dim = 2048

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward_features(self, x):
        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype
        return (
            torch.randn(b, 256, h // 4, w // 4, device=device, dtype=dtype) * self.scale,
            torch.randn(b, 512, h // 8, w // 8, device=device, dtype=dtype) * self.scale,
            torch.randn(b, 1024, h // 16, w // 16, device=device, dtype=dtype) * self.scale,
            torch.randn(b, 2048, h // 32, w // 32, device=device, dtype=dtype) * self.scale,
        )


def test_class_mix_batch_preserves_shapes_and_types():
    b, h, w = 2, 32, 32
    out = class_mix_batch(
        img_w=torch.randn(b, 3, h, w),
        img_s=torch.randn(b, 3, h, w),
        pseudo_label=torch.randint(0, 5, (b, h, w)),
        pseudo_logit=torch.rand(b, h, w),
        entropy=torch.rand(b, h, w),
        ignore_mask=torch.zeros(b, h, w, dtype=torch.long),
    )
    assert out["img_w"].shape == (b, 3, h, w)
    assert out["img_s"].shape == (b, 3, h, w)
    assert out["pseudo_label"].shape == (b, h, w)
    assert out["pseudo_label"].dtype == torch.long
    assert out["mix_mask"].shape == (b, h, w)




def test_apply_ignore_mask_to_labels_replaces_invalid_pseudo_pixels():
    labels = torch.tensor([[[1, 2], [3, 4]]])
    ignore_mask = torch.tensor([[[0, 255], [255, 0]]])
    masked = apply_ignore_mask_to_labels(labels, ignore_mask, ignore_index=99)
    assert torch.equal(masked, torch.tensor([[[1, 99], [99, 4]]]))


def test_build_entropy_targets_marks_unlabeled_ignore_pixels_invalid():
    mask_l = torch.tensor([[[0, 255], [1, 2]]])
    mixed_entropy = torch.tensor([[[0.3, 0.4], [0.5, 0.6]]])
    student_entropy = torch.zeros(2, 2, 2)
    ignore_mask = torch.tensor([[[0, 255], [0, 255]]])
    teacher_entropy_all, valid_entropy = build_entropy_targets(
        mask_l,
        mixed_entropy,
        student_entropy,
        ignore_mask,
        ignore_index=255,
    )
    assert teacher_entropy_all.shape == (2, 2, 2)
    assert torch.equal(valid_entropy[0], torch.tensor([[True, False], [True, True]]))
    assert torch.equal(valid_entropy[1], torch.tensor([[True, False], [True, False]]))
def test_generate_grid_mask_matches_requested_ratio_and_shape():
    mask = generate_grid_mask(2, 32, 32, mask_gap=8, mask_rate=0.25, device=torch.device("cpu"))
    assert mask.shape == (2, 1, 32, 32)
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})


def test_segmind_contrastive_loss_returns_scalar_and_updates_bank():
    bank = create_memory_bank(num_classes=3, proj_dim=8, bank_size=16, device=torch.device("cpu"))
    feat = torch.randn(2, 8, 8, 8)
    labels = torch.randint(0, 3, (2, 32, 32))
    prob = torch.softmax(torch.randn(2, 3, 32, 32), dim=1)
    loss = segmind_contrastive_loss(
        feat=feat,
        labels=labels,
        prob=prob,
        bank=bank,
        query_threshold=0.99,
        temperature=0.5,
        num_queries=4,
        num_negative=8,
    )
    assert loss.ndim == 0
    assert any(queue.shape[0] > 0 for queue in bank.queues)


def test_segmind_model_wraps_dpt_and_upernet(monkeypatch):
    monkeypatch.setattr(dpt_mod, "ResNet101Backbone", FakeResNet101Backbone)
    monkeypatch.setattr(upernet_mod, "ResNet101Backbone", FakeResNet101Backbone)

    dpt_model = dpt_mod.DPT(
        encoder_size="resnet101",
        nclass=3,
        features=64,
        out_channels=[256, 512, 1024, 2048],
        backbone_version="resnet",
    )
    dpt_wrapper = SegMindModel(dpt_model, nclass=3, project_dim=16)
    x = torch.randn(2, 3, 128, 128)
    out = dpt_wrapper(x, return_aux=True, mim_mask=torch.ones(2, 1, 128, 128))
    assert out["seg_logits"].shape == (2, 3, 128, 128)
    assert out["proj_feat"].shape[1] == 16
    assert out["recon_img"].shape[1] == 3

    uper_model = upernet_mod.UperNet(
        encoder_size="resnet101",
        nclass=3,
        fpn_channels=64,
        backbone_version="resnet",
    )
    uper_wrapper = SegMindModel(uper_model, nclass=3, project_dim=16)
    out = uper_wrapper(x, return_aux=True, mim_mask=torch.ones(2, 1, 128, 128))
    assert out["seg_logits"].shape == (2, 3, 128, 128)
    assert out["proj_feat"].shape[1] == 16


def test_segmind_source_uses_shared_helpers_and_validation_wrapper():
    text = (REPO_ROOT / "segmind.py").read_text(encoding="utf-8")
    assert 'wrap_ddp(model, logger=logger, rank=rank, save_path=args.save_path)' in text
    assert 'maybe_load_checkpoint(args, model, optimizer, model_ema=model_ema, logger=logger, rank=rank)' in text
    assert 'from util.validation import validation_cpu as shared_validation_cpu' in text
    assert 'return shared_validation_cpu(cfg, model, valid_loader)' in text
    assert 'strong_outputs = model(strong_inputs, return_aux=True)' in text
    assert 'masked_weak_inputs = weak_inputs * mim_mask' in text
    assert 'recon_outputs = model(masked_weak_inputs, mim_mask=mim_mask, return_aux=True)' in text
    assert 'feat=strong_outputs["proj_feat"]' in text
