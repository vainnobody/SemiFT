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

import dwl as dwl_trainer
from model.semseg import dpt as dpt_mod
from model.semseg import dpt_dwl as dpt_dwl_mod
from model.semseg import upernet as upernet_mod
from model.semseg import upernet_dwl as upernet_dwl_mod


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
    monkeypatch.setattr(dpt_dwl_mod, "DPT", dpt_mod.DPT)
    monkeypatch.setattr(upernet_mod, "ResNet101Backbone", FakeResNet101Backbone)


def test_dwl_source_matches_official_training_flow_expectations():
    source = (REPO_ROOT / "dwl.py").read_text()

    assert "model_ema" not in source
    assert "cutmix" not in source.lower()
    assert "(wgt_u + 1) / 2" not in source
    assert "return_pseudo_pred=True" in source
    assert "loss = loss_x + loss_u_s" in source
    assert "def infinite_loader(loader):" in source
    assert "cycle(trainloader_l)" not in source


def test_dpt_dwl_returns_main_and_pseudo_predictions():
    model = dpt_dwl_mod.DPT_DWL(
        encoder_size="resnet101",
        nclass=3,
        features=64,
        out_channels=[256, 512, 1024, 2048],
        backbone_version="resnet",
    )
    x = torch.randn(2, 3, 128, 128)
    pred, pseudo_pred = model(x, return_pseudo_pred=True)
    assert pred.shape == (2, 3, 128, 128)
    assert pseudo_pred.shape == pred.shape


def test_upernet_dwl_returns_main_and_pseudo_predictions():
    model = upernet_dwl_mod.UPerNet_DWL(
        encoder_size="resnet101",
        nclass=4,
        fpn_channels=64,
        backbone_version="resnet",
    )
    x = torch.randn(2, 3, 128, 128)
    pred, pseudo_pred = model(x, return_pseudo_pred=True)
    assert pred.shape == (2, 4, 128, 128)
    assert pseudo_pred.shape == pred.shape


def test_transfer_pseudo_head_updates_dpt_head_weights():
    model = dpt_dwl_mod.DPT_DWL(
        encoder_size="resnet101",
        nclass=2,
        features=64,
        out_channels=[256, 512, 1024, 2048],
        backbone_version="resnet",
    )
    for param in model.head.parameters():
        param.data.zero_()
    for param in model.pseudo_head.parameters():
        param.data.fill_(2.0)

    dwl_trainer.transfer_pseudo_head(model, 0.25)

    first_param = next(model.head.parameters())
    assert torch.allclose(first_param, torch.full_like(first_param, 0.5))


def test_transfer_pseudo_head_updates_upernet_classifier_weights():
    model = upernet_dwl_mod.UPerNet_DWL(
        encoder_size="resnet101",
        nclass=2,
        fpn_channels=64,
        backbone_version="resnet",
    )
    model.decoder.classifier.weight.data.zero_()
    model.decoder.classifier.bias.data.zero_()
    model.pseudo_classifier.weight.data.fill_(4.0)
    model.pseudo_classifier.bias.data.fill_(4.0)

    dwl_trainer.transfer_pseudo_head(model, 0.5)

    assert torch.allclose(
        model.decoder.classifier.weight,
        torch.full_like(model.decoder.classifier.weight, 2.0),
    )
    assert torch.allclose(
        model.decoder.classifier.bias,
        torch.full_like(model.decoder.classifier.bias, 2.0),
    )
