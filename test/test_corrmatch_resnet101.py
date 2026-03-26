import inspect
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

from model.semseg import corrmatch as corrmatch_mod
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
    monkeypatch.setattr(corrmatch_mod, "ResNet101Backbone", FakeResNet101Backbone, raising=False)


def test_dpt_corrmatch_supports_resnet101_need_fp():
    model = corrmatch_mod.DPT_CorrMatch(
        encoder_size="resnet101",
        nclass=3,
        features=64,
        out_channels=[256, 512, 1024, 2048],
        backbone_version="resnet",
    )
    x = torch.randn(2, 3, 128, 128)
    outputs = model(x, need_fp=True, use_corr=True)

    assert set(outputs) == {"out", "out_fp", "corr_out", "corr_map"}
    assert outputs["out"].shape == (2, 3, 128, 128)
    assert outputs["out_fp"].shape == (2, 3, 128, 128)
    assert outputs["corr_out"].shape == (2, 3, 128, 128)
    assert outputs["corr_map"].dtype == torch.bool
    assert outputs["corr_map"].shape[0] == 2
    assert outputs["corr_map"].shape[-2:] == (128, 128)


def test_upernet_corrmatch_supports_resnet101_need_fp():
    model = corrmatch_mod.UPerNet_CorrMatch(
        encoder_size="resnet101",
        nclass=4,
        fpn_channels=64,
        backbone_version="resnet",
    )
    x = torch.randn(2, 3, 128, 128)
    outputs = model(x, need_fp=True, use_corr=True)

    assert set(outputs) == {"out", "out_fp", "corr_out", "corr_map"}
    assert outputs["out"].shape == (2, 4, 128, 128)
    assert outputs["out_fp"].shape == (2, 4, 128, 128)
    assert outputs["corr_out"].shape == (2, 4, 128, 128)
    assert outputs["corr_map"].dtype == torch.bool
    assert outputs["corr_map"].shape[0] == 2
    assert outputs["corr_map"].shape[-2:] == (128, 128)


def test_corrmatch_script_uses_dynamic_threshold_and_single_strong_branch():
    import corrmatch as trainer

    source = inspect.getsource(trainer)
    assert "ThreshController" in source
    assert 'cfg.get("thresh_init", 0.85)' in source
    assert "zip(trainloader_l, trainloader_u, trainloader_u)" in source
    assert "img_u_s1" in source
    assert "pred_u_s1_corr" in source
    assert "apply_region_propagation" in source
