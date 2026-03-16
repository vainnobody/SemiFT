import inspect

import torch
import torchvision.transforms.functional as TF

from fixmatch_rgcrv6 import binarize_valid_mask
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
    valid_mask = binarize_valid_mask(valid_mask.squeeze(1).float())

    assert set(torch.unique(valid_mask).tolist()).issubset({0.0, 1.0})


def test_rgcrv6_keeps_rvs_branch_separate_from_cutmix_branch():
    import fixmatch_rgcrv6 as rgcrv6

    source = inspect.getsource(rgcrv6)

    assert "img_u_rvs" in source
    assert "pred_u_rvs = pred_u_s2" not in source
    assert "torch.cat((img_u_s1, img_u_rvs)), feature_perturb=feature_perturb" in source
    assert "img_u_rvs[cutmix_box2" not in source


def test_rgcrv6_uses_050025025_loss_weights():
    import fixmatch_rgcrv6 as rgcrv6
    source = inspect.getsource(rgcrv6)

    assert "loss_x * 0.5" in source
    assert "loss_u_s1 * 0.25" in source
    assert "loss_u_rvs * 0.25" in source
    assert "+ loss_u_rvs * 0.25\n            ) / 2.0" not in source
