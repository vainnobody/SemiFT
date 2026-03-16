import sys
import types
from types import SimpleNamespace

stub_tb = types.ModuleType("torch.utils.tensorboard")
class StubSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass
    def add_scalar(self, *args, **kwargs):
        pass
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
class StubAdaptModel:
    def __init__(self, config, model):
        self.config = config
        self.backbone = model.backbone
        self.head = model.head
        self.model = model
        self._params = list(model.named_parameters())
    def named_parameters(self):
        return iter(self._params)
stub_peft_semift.SemiFTConfig = StubSemiFTConfig
stub_peft_semift.AdaptModel = StubAdaptModel
sys.modules.setdefault("peft", stub_peft)
sys.modules.setdefault("peft.tuners", stub_peft_tuners)
sys.modules["peft.tuners.semift"] = stub_peft_semift

import torch
import scalematch_peft
from model.semseg import dpt_scalematch as scalematch_model


class DummyBackbone(torch.nn.Module):
    def __init__(self, model_name="small"):
        super().__init__()
        self.embed_dim = 32
        self.patch_size = 16
        self.weight = torch.nn.Parameter(torch.ones(1))
    def get_intermediate_layers(self, x, idx):
        b, _, h, w = x.shape
        patch_h = h // self.patch_size
        patch_w = w // self.patch_size
        n = patch_h * patch_w
        d = self.embed_dim
        base = torch.arange(n, dtype=x.dtype, device=x.device).view(1, n, 1)
        return tuple(base.repeat(b, 1, d) for _ in idx)


def make_args(**kwargs):
    defaults = {
        "peft_method": None,
        "peft_target_modules": None,
        "freeze_backbone": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_resolve_peft_cfg_for_scalematch_from_yaml():
    cfg = {"nclass": 5, "peft": {"method": "lora", "target_modules": ["attn"], "freeze_backbone": False}}
    peft_cfg = scalematch_peft.resolve_peft_cfg(cfg, make_args())
    assert peft_cfg["method"] == "lora"
    assert peft_cfg["target_modules"] == ["attn"]
    assert peft_cfg["freeze_backbone"] is False


def test_scalematch_peft_build_model_wraps_dpt_scalematch(monkeypatch):
    monkeypatch.setattr(scalematch_model, "DINOv2", DummyBackbone)
    monkeypatch.setattr(scalematch_model, "DINOv3", DummyBackbone)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    cfg = {"backbone": "dinov2_small", "nclass": 5, "model": "dpt", "peft": {"method": "lora"}}
    peft_cfg = scalematch_peft.resolve_peft_cfg(cfg, make_args())
    model, load_result, ckpt = scalematch_peft.build_model(cfg, peft_cfg)
    assert isinstance(model, StubAdaptModel)
    assert model.backbone.patch_size == 16
    assert ckpt.endswith("dinov2_small.pth")


def test_scalematch_peft_optimizer_only_uses_trainable_params():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 2)
            self.backbone.weight.requires_grad = False
    model = ToyModel()
    optim = scalematch_peft.build_optimizer(model, {"lr": 1e-4, "lr_multi": 10.0})
    assert len(optim.param_groups) == 2
    assert all(p.requires_grad for p in optim.param_groups[0]["params"] + optim.param_groups[1]["params"])


def test_scalematch_peft_source_uses_scalematch_and_peft_contracts():
    source = open("scalematch_peft.py", "r", encoding="utf-8").read()
    assert "DPT_ScaleMatch" in source
    assert "build_peft_config" in source
    assert "find_unused_parameters=False" in source
    assert "epoch_repeat_factor" in source
