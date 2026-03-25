import inspect
import sys
import types
from pathlib import Path


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

import unimatch as trainer


def test_unimatch_uses_three_loaders_and_feature_perturb_branch():
    source = inspect.getsource(trainer)

    assert "zip(trainloader_l, trainloader_u, trainloader_u_mix)" in source
    assert "preds, preds_fp = model(torch.cat((img_x, img_u_w)), need_fp=True)" in source
    assert "pred_u_w_fp = preds_fp[num_lb:]" in source


def test_unimatch_uses_official_dual_strong_and_mix_donor_logic():
    source = inspect.getsource(trainer)

    assert "img_u_s1_mix" in source
    assert "img_u_s2_mix" in source
    assert "pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)" in source
    assert "mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]" in source
    assert "mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]" in source
    assert "model_ema" not in source
    assert "comp_drop=True" not in source


def test_unimatch_uses_expected_loss_weights_and_semi_rs_compatibility_switch():
    source = inspect.getsource(trainer)

    assert "loss_u_s1 * 0.25" in source
    assert "loss_u_s2 * 0.25" in source
    assert "loss_u_w_fp * 0.5" in source
    assert 'or "semi_rs"' in source
    assert "RemoteSemiDataset" in source
