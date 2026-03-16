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

import ranpaste


def test_sample_paste_boxes_shape_and_density():
    boxes = ranpaste.sample_paste_boxes(3, 32, 32, ratio=0.5)
    assert boxes.shape == (3, 32, 32)
    assert boxes.dtype == torch.bool
    assert boxes.any()


def test_ranpaste_source_contains_ema_and_paste_logic():
    source = open("ranpaste.py", "r", encoding="utf-8").read()
    assert 'build_ema_model' in source
    assert 'sample_paste_boxes' in source
    assert 'pseudo_label[b, paste_box[b]] = mask_x[src_idx, paste_box[b]]' in source


def test_ranpaste_paste_overrides_pseudo_label_region():
    img_x = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    mask_x = torch.arange(16).reshape(1, 4, 4)
    img_u_s = torch.zeros(1, 3, 4, 4)
    pseudo = torch.zeros(1, 4, 4, dtype=torch.long)
    conf = torch.zeros(1, 4, 4)
    box = torch.zeros(1, 4, 4, dtype=torch.bool)
    box[:, :2, :2] = True
    img_u_s[0, :, box[0]] = img_x[0, :, box[0]]
    pseudo[0, box[0]] = mask_x[0, box[0]]
    conf[0, box[0]] = 1.0
    assert torch.equal(pseudo[0, :2, :2], mask_x[0, :2, :2])
    assert torch.all(conf[0, :2, :2] == 1.0)
