import sys
import types

import torch

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

import scalematch
from model.semseg import dpt_scalematch as scalematch_model


class DummyBackbone(torch.nn.Module):
    def __init__(self, model_name="small"):
        super().__init__()
        self.model_name = model_name
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


def install_dummy_backbone(monkeypatch):
    monkeypatch.setattr(scalematch_model, "DINOv2", DummyBackbone)
    monkeypatch.setattr(scalematch_model, "DINOv3", DummyBackbone)


def test_get_scalematch_dataset_cls_routes_expected_loaders():
    cls, name = scalematch.get_scalematch_dataset_cls("pascal")
    assert cls.__module__ == "dataset.semi"
    assert name == "semi"

    cls, name = scalematch.get_scalematch_dataset_cls("potsdam")
    assert cls.__module__ == "dataset.semi_rs"
    assert name == "semi_rs"


def test_get_scalematch_dataset_cls_rejects_unknown_dataset():
    try:
        scalematch.get_scalematch_dataset_cls("unknown")
    except ValueError as exc:
        assert "Unsupported dataset" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown dataset")


def test_resize_x_keeps_patch_alignment():
    x = torch.randn(2, 3, 65, 97)
    y = scalematch_model.resize_x(x, 1.5, patch_size=16)
    assert y.shape[-2] % 16 == 0
    assert y.shape[-1] % 16 == 0
    assert y.shape[-2] >= 16
    assert y.shape[-1] >= 16


def test_scalematch_forward_contract(monkeypatch):
    install_dummy_backbone(monkeypatch)
    model = scalematch_model.DPT_ScaleMatch(
        encoder_size="small",
        nclass=5,
        features=32,
        out_channels=[8, 16, 32, 32],
        backbone_version="dinov2",
    )

    x = torch.randn(2, 3, 64, 64)
    single = model(x, scale_factor=None)
    assert isinstance(single, torch.Tensor)
    assert single.shape == (2, 5, 64, 64)

    multi = model(x, scale_factor=0.5, feature_scale=1.0)
    assert set(multi.keys()) == {"pred_joint", "pred_ori", "pred_fp", "pred_size"}
    for value in multi.values():
        assert isinstance(value, torch.Tensor)
        assert value.shape == (2, 5, 64, 64)


def test_lock_backbone_freezes_parameters(monkeypatch):
    install_dummy_backbone(monkeypatch)
    model = scalematch_model.DPT_ScaleMatch(
        encoder_size="small",
        nclass=3,
        features=32,
        out_channels=[8, 16, 32, 32],
        backbone_version="dinov2",
    )
    assert any(p.requires_grad for p in model.backbone.parameters())
    model.lock_backbone()
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_scalematch_trainer_uses_safe_ddp_settings():
    source = open("scalematch.py", "r", encoding="utf-8").read()
    assert "find_unused_parameters=True" in source
    assert "static_graph=True" not in source
    assert "model.no_sync()" not in source


def test_scalematch_teacher_pseudo_labels_run_in_eval_block():
    source = open("scalematch.py", "r", encoding="utf-8").read()
    start = source.index("model.eval()")
    end = source.index("model.train()", start)
    block = source[start:end]
    assert "pred_u_w_mix = model.module" in block
    assert "pred_teacher_for_strong = model.module" in block


def test_scalematch_uses_adamw_optimizer():
    source = open("scalematch.py", "r", encoding="utf-8").read()
    assert "from torch.optim import AdamW" in source
    assert "optimizer = AdamW(" in source
    assert "from torch.optim import SGD" not in source


def test_scalematch_logs_backbone_load_result():
    source = open("scalematch.py", "r", encoding="utf-8").read()
    assert "backbone_ckpt_path" in source
    assert 'map_location="cpu"' in source
    assert "load_result = model.backbone.load_state_dict(state_dict)" in source
    assert "missing_keys" in source
    assert "unexpected_keys" in source
