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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scalematch
from dataset.semi_rs import SemiDataset as RemoteSemiDataset
from model.semseg import dpt as dpt_mod
from model.semseg import upernet as upernet_mod


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


@pytest.fixture(autouse=True)
def patch_fake_resnet(monkeypatch):
    monkeypatch.setattr(dpt_mod, "ResNet101Backbone", FakeResNet101Backbone)
    monkeypatch.setattr(upernet_mod, "ResNet101Backbone", FakeResNet101Backbone)


def test_scalematch_recipe_uses_defaults():
    recipe = scalematch.get_scalematch_recipe({"dataset": "vaihingen"})
    assert recipe["conf_thresh"] == 0.95
    assert recipe["img_scales"] == [0.5, 0.75, 1.0, 1.25]
    assert recipe["feat_l_scales"] == [1.0, 1.25, 1.5]
    assert recipe["warm_up"] == 10


def test_get_scalematch_dataset_cls_is_unified_to_semi_rs():
    dataset_cls, loader_name = scalematch.get_scalematch_dataset_cls("pascal")
    assert dataset_cls is RemoteSemiDataset
    assert loader_name == "semi_rs"


def test_build_scalematch_model_supports_generic_dpt_and_upernet():
    dpt_model, backbone_version = scalematch.build_scalematch_model(
        {"backbone": "resnet101", "nclass": 3, "model": "dpt"}
    )
    assert isinstance(dpt_model, dpt_mod.DPT)
    assert dpt_model.enable_scalematch is True
    assert backbone_version == "resnet"

    uper_model, backbone_version = scalematch.build_scalematch_model(
        {"backbone": "resnet101", "nclass": 3, "model": "upernet"}
    )
    assert isinstance(uper_model, upernet_mod.UperNet)
    assert uper_model.enable_scalematch is True
    assert backbone_version == "resnet"


def test_build_scalematch_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="supports only 'dpt' and 'upernet'"):
        scalematch.build_scalematch_model(
            {"backbone": "resnet101", "nclass": 3, "model": "unknown"}
        )


def test_dpt_scalematch_forward_outputs_expected_shapes():
    model = dpt_mod.DPT(
        encoder_size="resnet101",
        nclass=3,
        features=64,
        out_channels=[256, 512, 1024, 2048],
        backbone_version="resnet",
        enable_scalematch=True,
    )
    x = torch.randn(2, 3, 128, 128)

    logits = model(x)
    assert logits.shape == (2, 3, 128, 128)

    hi = model(x, scale_factor=1.5, feature_scale=0.75)
    lo = model(x, scale_factor=0.5, feature_scale=1.25)
    for out in (hi, lo):
        assert set(out) == {"pred_joint", "pred_ori", "pred_fp", "pred_size"}
        for value in out.values():
            assert value.shape == (2, 3, 128, 128)


def test_upernet_scalematch_forward_outputs_expected_shapes():
    model = upernet_mod.UperNet(
        encoder_size="resnet101",
        nclass=3,
        fpn_channels=64,
        backbone_version="resnet",
        enable_scalematch=True,
    )
    x = torch.randn(2, 3, 128, 128)

    logits = model(x)
    assert logits.shape == (2, 3, 128, 128)

    hi = model(x, scale_factor=1.5, feature_scale=0.75)
    lo = model(x, scale_factor=0.5, feature_scale=1.25)
    for out in (hi, lo):
        assert set(out) == {"pred_joint", "pred_ori", "pred_fp", "pred_size"}
        for value in out.values():
            assert value.shape == (2, 3, 128, 128)


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
