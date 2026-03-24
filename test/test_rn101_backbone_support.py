import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

stub_tb = types.ModuleType("torch.utils.tensorboard")


class StubSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        return None


stub_tb.SummaryWriter = StubSummaryWriter
sys.modules.setdefault("torch.utils.tensorboard", stub_tb)

from model.semseg import dpt as dpt_mod
from model.semseg import dpt_segmind as dpt_segmind_mod
from model.semseg import upernet as upernet_mod
from model.backbone.resnet import ResNet101Backbone
from util import ssl_method_utils as ssl_utils


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


def test_parse_backbone_spec_supports_rn101_aliases():
    for name in ("resnet101", "rn101", "rn-101", "RN-101"):
        info = ssl_utils.parse_backbone_spec(name)
        assert info["canonical_name"] == "resnet101"
        assert info["family"] == "resnet"
        assert info["size"] == "resnet101"


def test_rn101_uses_default_pretrained_checkpoint(monkeypatch):
    cfg = {"backbone": "resnet101", "nclass": 5}
    monkeypatch.setattr(Path, "exists", lambda self: str(self).endswith("pretrained/resnet101.pth"))
    assert ssl_utils.get_backbone_checkpoint_path(cfg).endswith(
        "pretrained/resnet101.pth"
    )


def test_rn101_prefers_explicit_checkpoint_path():
    cfg = {
        "backbone": "resnet101",
        "nclass": 5,
        "backbone_ckpt": "/tmp/custom_resnet101.pth",
    }
    assert ssl_utils.get_backbone_checkpoint_path(cfg) == "/tmp/custom_resnet101.pth"


def test_rn101_raises_when_no_checkpoint_is_available(monkeypatch):
    cfg = {"backbone": "resnet101", "nclass": 5}
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(ValueError, match="pretrained/resnet101.pth"):
        ssl_utils.get_backbone_checkpoint_path(cfg)


def test_resnet_backbone_accepts_official_torchvision_fc_keys():
    backbone = ResNet101Backbone()
    state_dict = backbone.state_dict()
    state_dict["fc.weight"] = torch.randn(1000, 2048)
    state_dict["fc.bias"] = torch.randn(1000)

    load_result = backbone.load_state_dict(state_dict, strict=False)

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []


def test_dpt_supports_resnet101_backbone():
    model = dpt_mod.DPT(
        encoder_size="resnet101",
        nclass=3,
        features=64,
        out_channels=[256, 512, 1024, 2048],
        backbone_version="resnet",
    )
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 3, 128, 128)


def test_upernet_supports_resnet101_backbone():
    model = upernet_mod.UperNet(
        encoder_size="resnet101",
        nclass=4,
        fpn_channels=64,
        backbone_version="resnet",
    )
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 4, 128, 128)


def test_upernet_supports_resnet101_need_fp():
    model = upernet_mod.UperNet(
        encoder_size="resnet101",
        nclass=4,
        fpn_channels=64,
        backbone_version="resnet",
    )
    x = torch.randn(2, 3, 128, 128)
    y, y_fp = model(x, need_fp=True)
    assert y.shape == (2, 4, 128, 128)
    assert y_fp.shape == (2, 4, 128, 128)


def test_dpt_supports_resnet101_need_fp():
    model = dpt_mod.DPT(
        encoder_size="resnet101",
        nclass=3,
        features=64,
        out_channels=[256, 512, 1024, 2048],
        backbone_version="resnet",
    )
    x = torch.randn(2, 3, 128, 128)
    y, y_fp = model(x, need_fp=True)
    assert y.shape == (2, 3, 128, 128)
    assert y_fp.shape == (2, 3, 128, 128)


def test_dpt_segmind_supports_resnet101_aux_outputs():
    model = dpt_segmind_mod.DPT_SegMind(
        encoder_size="resnet101",
        nclass=5,
        features=64,
        out_channels=[256, 512, 1024, 2048],
        backbone_version="resnet",
        proj_dim=16,
    )
    x = torch.randn(2, 3, 128, 128)
    outputs = model(
        x,
        return_proj=True,
        return_reconstruction=True,
        reconstruction_mask=torch.ones(2, 1, 128, 128),
    )
    assert outputs["out"].shape == (2, 5, 128, 128)
    assert outputs["proj_feat"].shape[-2:] == (64, 64)
    assert outputs["recon"].shape == (2, 3, 128, 128)


def test_scalematch_models_support_resnet101_backbone():
    x = torch.randn(2, 3, 128, 128)

    dpt_model = dpt_mod.DPT(
        encoder_size="resnet101",
        nclass=3,
        features=64,
        out_channels=[256, 512, 1024, 2048],
        backbone_version="resnet",
        enable_scalematch=True,
    )
    y = dpt_model(x)
    assert y.shape == (2, 3, 128, 128)
    scale_out = dpt_model(x, scale_factor=1.25, feature_scale=1.25)
    assert set(scale_out) == {"pred_joint", "pred_ori", "pred_fp", "pred_size"}

    uper_model = upernet_mod.UperNet(
        encoder_size="resnet101",
        nclass=3,
        fpn_channels=64,
        backbone_version="resnet",
        enable_scalematch=True,
    )
    y = uper_model(x)
    assert y.shape == (2, 3, 128, 128)
    scale_out = uper_model(x, scale_factor=1.25, feature_scale=1.25)
    assert set(scale_out) == {"pred_joint", "pred_ori", "pred_fp", "pred_size"}
