# Original copyright:
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Github repo: https://github.com/google-research/semivl/blob/main/utils/train_utils.py
# Modified for SemiFT project

import numpy as np
import torch


def cutmix_img_(img, img_mix, cutmix_box):
    """In-place CutMix operation on images.

    Args:
        img: Target image tensor (B, C, H, W)
        img_mix: Source image tensor to mix from (B, C, H, W)
        cutmix_box: Binary mask indicating regions to mix (B, H, W)
    """
    img[cutmix_box.unsqueeze(1).expand(img.shape) == 1] = img_mix[
        cutmix_box.unsqueeze(1).expand(img.shape) == 1
    ]


def cutmix_mask(mask, mask_mix, cutmix_box):
    """CutMix operation on masks.

    Args:
        mask: Target mask tensor (B, H, W)
        mask_mix: Source mask tensor to mix from (B, H, W)
        cutmix_box: Binary mask indicating regions to mix (B, H, W)

    Returns:
        Mixed mask tensor
    """
    cutmixed = mask.clone()
    cutmixed[cutmix_box == 1] = mask_mix[cutmix_box == 1]
    return cutmixed


def confidence_weighted_loss(
    loss, conf_map, ignore_mask, ignore_index, conf_thresh=0.95, conf_mode="pixelwise"
):
    """Compute confidence-weighted loss for semi-supervised learning.

    Args:
        loss: Per-pixel loss tensor (B, H, W)
        conf_map: Confidence map from teacher predictions (B, H, W)
        ignore_mask: Mask indicating pixels to ignore (B, H, W), 255 = ignore
        conf_thresh: Confidence threshold for pseudo-labels
        conf_mode: Weighting mode - 'pixelwise', 'pixelratio', or 'pixelavg'

    Returns:
        Scalar loss value
    """
    # assert loss.dim() == 3
    # assert conf_map.dim() == 3
    # assert ignore_mask.dim() == 3
    valid_mask = ignore_mask != ignore_index
    sum_pixels = dict(dim=(1, 2), keepdim=True)

    if conf_mode == "pixelwise":
        loss = loss * ((conf_map >= conf_thresh) & valid_mask)
        loss = loss.sum() / valid_mask.sum().clamp(min=1.0)
    elif conf_mode == "pixelratio":
        ratio_high_conf = ((conf_map >= conf_thresh) & valid_mask).sum(
            **sum_pixels
        ) / valid_mask.sum(**sum_pixels).clamp(min=1.0)
        loss = loss * ratio_high_conf
        loss = loss.sum() / valid_mask.sum().clamp(min=1.0)
    elif conf_mode == "pixelavg":
        avg_conf = (conf_map * valid_mask).sum(**sum_pixels) / valid_mask.sum(
            **sum_pixels
        ).clamp(min=1.0)
        loss = loss.sum() * avg_conf
        loss = loss.sum() / valid_mask.sum().clamp(min=1.0)
    else:
        raise ValueError(f"Unknown conf_mode: {conf_mode}")
    return loss


class DictAverageMeter:
    """Average meter that can track multiple metrics using a dictionary."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avgs = {}
        self.sums = {}
        self.counts = {}

    def update(self, vals):
        """Update metrics with new values.

        Args:
            vals: Dictionary of metric names to values
        """
        for k, v in vals.items():
            if torch.is_tensor(v):
                v = v.detach()
            if k not in self.sums:
                self.sums[k] = 0
                self.counts[k] = 0
            self.sums[k] += v
            self.counts[k] += 1
            self.avgs[k] = torch.true_divide(self.sums[k], self.counts[k])

    def __str__(self):
        s = []
        for k, v in self.avgs.items():
            if torch.is_tensor(v):
                s.append(f"{k}: {v.item():.3f}")
            else:
                s.append(f"{k}: {v:.3f}")
        return ", ".join(s)


def generate_lambda_schedule(epochs, total_epochs, warmup_epochs):
    """Generate lambda value for loss weighting schedule.

    Args:
        epochs: Current epoch
        total_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs

    Returns:
        Lambda value for current epoch
    """
    if epochs < warmup_epochs:
        lambda_values = epochs / warmup_epochs
    else:
        lambda_values = 1 - (epochs - warmup_epochs) / (total_epochs - warmup_epochs)
    return lambda_values
