import sys
import types
from pathlib import Path
from types import SimpleNamespace

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

stub_peft = types.ModuleType("peft")
stub_peft.__path__ = []
stub_peft_tuners = types.ModuleType("peft.tuners")
stub_peft_tuners.__path__ = []
stub_peft_semift = types.ModuleType("peft.tuners.semift")


class StubSemiFTConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class StubAdaptModel(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


stub_peft_semift.SemiFTConfig = StubSemiFTConfig
stub_peft_semift.AdaptModel = StubAdaptModel
sys.modules.setdefault("peft", stub_peft)
sys.modules.setdefault("peft.tuners", stub_peft_tuners)
sys.modules["peft.tuners.semift"] = stub_peft_semift

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scalematch_peft
from model.semseg import dpt_scalematch
from model.semseg import upernet_scalematch


class FakeBackbone(torch.nn.Module):
    def __init__(self, model_name="small"):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = 8
        self.patch_size = 14
        self.proj = torch.nn.Linear(8, 8)

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
    monkeypatch.setattr(
        scalematch_peft.torch,
        "load",
        lambda *args, **kwargs: FakeBackbone().state_dict(),
    )


def make_args(**kwargs):
    defaults = {
        "peft_method": None,
        "peft_target_modules": None,
        "freeze_backbone": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize("model_name", ["dpt", "upernet"])
def test_build_model_wraps_scalematch_with_peft(model_name):
    cfg = {
        "backbone": "dinov2_small",
        "nclass": 3,
        "model": model_name,
        "fpn_channels": 16,
    }
    peft_cfg = scalematch_peft.resolve_peft_cfg(
        {**cfg, "peft": {"method": "semift", "freeze_backbone": True}},
        make_args(),
    )

    model, backbone_version = scalematch_peft.build_model(cfg, peft_cfg)

    assert backbone_version == "dinov2"
    assert isinstance(model, StubAdaptModel)
    assert model.peft_config.method == "semift"
    assert all(not p.requires_grad for p in model.backbone.parameters())


@pytest.mark.parametrize("freeze_backbone", [True, False])
def test_build_optimizer_uses_only_trainable_params(freeze_backbone):
    cfg = {
        "backbone": "dinov2_small",
        "nclass": 3,
        "model": "dpt",
        "lr": 1e-4,
        "lr_multi": 10.0,
    }
    peft_cfg = scalematch_peft.resolve_peft_cfg(
        {
            **cfg,
            "peft": {
                "method": "semift",
                "freeze_backbone": freeze_backbone,
            },
        },
        make_args(),
    )
    model, _ = scalematch_peft.build_model(cfg, peft_cfg)

    optimizer = scalematch_peft.build_optimizer(model, cfg)

    backbone_group = optimizer.param_groups[0]["params"]
    non_backbone_group = optimizer.param_groups[1]["params"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(cfg["lr"])
    assert optimizer.param_groups[1]["lr"] == pytest.approx(cfg["lr"] * cfg["lr_multi"])

    if freeze_backbone:
        assert len(backbone_group) == 0
    else:
        assert len(backbone_group) > 0
    assert len(non_backbone_group) > 0


class DummyDDP:
    def __init__(self):
        self.called = []

    def _set_static_graph(self):
        self.called.append("private")


class DummyDDPPublic:
    def __init__(self):
        self.called = []

    def set_static_graph(self):
        self.called.append("public")


def test_enable_ddp_static_graph_prefers_private_method():
    model = DummyDDP()
    scalematch_peft.enable_ddp_static_graph(model)
    assert model.called == ["private"]


def test_enable_ddp_static_graph_supports_public_method():
    model = DummyDDPPublic()
    scalematch_peft.enable_ddp_static_graph(model)
    assert model.called == ["public"]
