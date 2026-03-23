import inspect
import sys
import types


stub_tb = types.ModuleType("torch.utils.tensorboard")


class StubSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass


stub_tb.SummaryWriter = StubSummaryWriter
sys.modules.setdefault("torch.utils.tensorboard", stub_tb)

import unimatch as trainer


def test_unimatch_uses_supervised_style_no_ema_unimatch_flow():
    source = inspect.getsource(trainer)

    assert 'from dataset.semi_rs import SemiDataset' in source
    assert 'from model.semseg.dpt import DPT' in source
    assert 'DPT_UniMatch' not in source
    assert 'model_ema' not in source
    assert 'preds, preds_fp = model(torch.cat((img_x, img_u_w)), need_fp=True)' in source
    assert 'pred_u_w = pred_u_w.detach()' in source
    assert 'loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5' in source
