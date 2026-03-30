import inspect
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

import scalematch as trainer
from model.semseg.scalematch import ScaleMatchModel


class DummyBackbone(torch.nn.Module):
    patch_size = 14

    def forward(self, x):
        return x


class DummySegModel(torch.nn.Module):
    def __init__(self, nclass=5):
        super().__init__()
        self.backbone = DummyBackbone()
        self.head = torch.nn.Conv2d(nclass, nclass, 1)
        self.nclass = nclass

    def lock_backbone(self):
        return None

    def forward(self, x, need_fp=False, **kwargs):
        logits = x[:, : self.nclass]
        if need_fp:
            return logits, logits + 1
        return logits


def test_scalematch_wrapper_returns_expected_keys_and_shapes():
    model = ScaleMatchModel(DummySegModel(nclass=4), nclass=4)
    x = torch.randn(2, 4, 63, 71)
    plain_x = torch.randn(2, 4, 63, 71)

    out = model(x, scale_factor=1.5, feature_scale=0.75, plain_inputs=plain_x)

    assert set(out.keys()) >= {"pred_joint", "pred_ori", "pred_fp", "pred_size", "pred_plain", "out"}
    for key in ("pred_joint", "pred_ori", "pred_fp", "pred_size", "pred_plain", "out"):
        assert out[key].shape == (2, 4, 63, 71)


def test_scalematch_wrapper_supports_small_scale_and_plain_forward():
    model = ScaleMatchModel(DummySegModel(nclass=3), nclass=3)
    x = torch.randn(1, 3, 65, 67)

    plain = model(x, scale_factor=None)
    scaled = model(x, scale_factor=0.5)

    assert plain.shape == (1, 3, 65, 67)
    assert scaled["pred_joint"].shape == (1, 3, 65, 67)


def test_scalematch_uses_official_three_loader_recipe_and_joint_outputs():
    source = inspect.getsource(trainer)

    assert "loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)" in source
    assert 'pred["pred_joint"]' in source
    assert 'pred["pred_size"]' in source
    assert 'pred["pred_fp"]' in source
    assert "if epoch < cfg[\"warm_up\"]:" in source
    assert "loss_u_s1 * 0.25 + loss_u_size * 0.25 + loss_u_w_fp * 0.5" in source
