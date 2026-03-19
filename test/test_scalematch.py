import sys
import types
from pathlib import Path

import pytest
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scalematch
from model.semseg import dpt_scalematch


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


def build_test_model():
    return dpt_scalematch.DPT_ScaleMatch(
        encoder_size="small",
        nclass=3,
        features=16,
        out_channels=[8, 8, 8, 8],
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
    with pytest.raises(ValueError, match="supports only 'dpt'"):
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
