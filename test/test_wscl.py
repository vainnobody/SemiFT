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

import wscl


def test_wscl_source_uses_unimatchv2_style_cli_and_ddp():
    source = open("wscl.py", "r", encoding="utf-8").read()
    assert 'parser.add_argument("--config"' in source
    assert 'find_unused_parameters=True' in source
    assert 'maybe_load_checkpoint' in source


def test_wscl_aug_modes_preserve_shapes():
    b, h, w = 2, 32, 32
    conf = torch.rand(b, h, w)
    mask = torch.randint(0, 5, (b, h, w))
    s1 = torch.randn(b, 3, h, w)
    s2 = torch.randn(b, 3, h, w)
    for mode, fn in wscl.AUG_HANDLERS.items():
        conf_out, mask_out, img_out = fn(conf, mask, s1, s2)
        assert conf_out.shape == conf.shape
        assert mask_out.shape == mask.shape
        assert img_out.shape == s1.shape


def test_entropy_map_returns_hw_map():
    probs = torch.softmax(torch.randn(2, 5, 16, 16), dim=1)
    entropy = wscl.entropy_map(probs, 1)
    assert entropy.shape == (2, 16, 16)
    assert torch.all(entropy >= 0)
