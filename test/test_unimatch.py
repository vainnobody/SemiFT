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

stub_supervised = types.ModuleType("supervised")


def _validation_cpu(*args, **kwargs):
    return 0.0, []


stub_supervised.validation_cpu = _validation_cpu
sys.modules.setdefault("supervised", stub_supervised)

import unimatch as trainer


def test_unimatch_uses_explicit_need_fp_for_weak_branch():
    source = inspect.getsource(trainer)

    assert "preds, preds_fp = model(torch.cat((img_x, img_u_w)), need_fp=True)" in source
    assert "preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)" not in source
