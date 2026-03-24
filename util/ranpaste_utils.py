import torch


def apply_paste_mask(base, paste, paste_mask):
    """Paste `paste` into `base` where paste_mask == 1.

    Supports tensors shaped (N, C, H, W) or (N, H, W).
    """
    mixed = base.clone()
    if paste_mask.dim() == base.dim() - 1:
        expanded_mask = paste_mask.unsqueeze(1).expand_as(base)
    else:
        expanded_mask = paste_mask.expand_as(base)
    mixed[expanded_mask == 1] = paste[expanded_mask == 1]
    return mixed


@torch.no_grad()
def build_ranpaste_targets(
    mask_u_w,
    conf_u_w,
    ignore_mask,
    mask_x,
    paste_mask,
    conf_thresh,
    ignore_index=255,
):
    mixed_target = apply_paste_mask(mask_u_w, mask_x, paste_mask)

    pseudo_valid_mask = (conf_u_w >= conf_thresh) & (ignore_mask != 255)
    pasted_valid_mask = (paste_mask == 1) & (mask_x != ignore_index)
    valid_mask = pseudo_valid_mask.clone()
    valid_mask[pasted_valid_mask] = True
    return mixed_target, valid_mask


@torch.no_grad()
def build_ranpaste_images(img_u, img_x, paste_mask):
    return apply_paste_mask(img_u, img_x, paste_mask)
