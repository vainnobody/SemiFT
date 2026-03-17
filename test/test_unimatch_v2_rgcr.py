import inspect
import sys
import types

import torch
import torchvision.transforms.functional as TF

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

import unimatch_v2_rgcr as trainer
from model.semseg.rgcr_utils import scale_back


def test_scale_back_valid_mask_is_binary_after_binarization():
    size = 32
    pred_u_rvs = torch.randn(1, 3, size, size)
    mask_c = torch.ones(1, 1, size, size)
    mask_c = TF.rotate(
        mask_c,
        angle=37.0,
        interpolation=TF.InterpolationMode.NEAREST,
        fill=0,
    )
    box = torch.tensor([[0, 0, 0, 0, 1.0, 37.0]], dtype=torch.float32)

    _, valid_mask = scale_back(pred_u_rvs, mask_c, size, box)
    valid_mask = trainer.binarize_valid_mask(valid_mask.squeeze(1).float())

    assert set(torch.unique(valid_mask).tolist()).issubset({0.0, 1.0})


def test_unimatch_v2_rgcr_keeps_dual_strong_and_independent_rvs_branch():
    source = inspect.getsource(trainer)

    assert "torch.cat((img_u_s1, img_u_s2)), comp_drop=True" in source
    assert "pred_u_rvs = model(img_u_rvs)" in source
    assert "img_u_rvs[cutmix_box1" not in source
    assert "img_u_rvs[cutmix_box2" not in source
    assert "pred_u_rvs = pred_u_s2" not in source


def test_unimatch_v2_rgcr_uses_expected_loss_weights():
    source = inspect.getsource(trainer)

    assert "loss_x * 0.5" in source
    assert "loss_u_s1 / 6.0" in source
    assert "loss_u_s2 / 6.0" in source
    assert "loss_u_rvs / 6.0" in source
    assert ") / 2.0" not in source[source.index("loss = (") : source.index("torch.distributed.barrier()")]
