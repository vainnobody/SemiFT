import sys

import torch

sys.path.insert(0, "/Users/lanjie/Proj/SSL/SemiFT")

from util.ranpaste_utils import build_ranpaste_images, build_ranpaste_targets


def test_build_ranpaste_images_pastes_labeled_region_into_unlabeled_image():
    img_u = torch.zeros(1, 3, 4, 4)
    img_x = torch.ones(1, 3, 4, 4)
    paste_mask = torch.zeros(1, 4, 4)
    paste_mask[:, 1:3, 1:3] = 1

    mixed = build_ranpaste_images(img_u, img_x, paste_mask)

    assert torch.all(mixed[:, :, 1:3, 1:3] == 1)
    assert torch.all(mixed[:, :, 0, :] == 0)
    assert torch.all(mixed[:, :, :, 0] == 0)


def test_build_ranpaste_targets_uses_gt_inside_paste_and_pseudo_outside():
    mask_u_w = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]])
    conf_u_w = torch.tensor([[[0.9, 0.2, 0.95, 0.1], [0.8, 0.7, 0.96, 0.3], [0.9, 0.4, 0.2, 0.99], [0.8, 0.85, 0.95, 0.97]]])
    ignore_mask = torch.zeros_like(mask_u_w)
    ignore_mask[:, 3, 0] = 255
    mask_x = torch.full_like(mask_u_w, 4)
    paste_mask = torch.zeros_like(mask_u_w)
    paste_mask[:, :2, :2] = 1

    mixed_target, valid_mask = build_ranpaste_targets(
        mask_u_w, conf_u_w, ignore_mask, mask_x, paste_mask, conf_thresh=0.85
    )

    assert torch.all(mixed_target[:, :2, :2] == 4)
    assert torch.equal(mixed_target[:, 2:, 2:], mask_u_w[:, 2:, 2:])

    assert torch.all(valid_mask[:, :2, :2])
    assert valid_mask[:, 0, 2].item() is True
    assert valid_mask[:, 2, 1].item() is False
    assert valid_mask[:, 3, 0].item() is False
