"""
Focal Loss implementation for SemiFT project.
Adapted from dinov3_segmentation/loss.py with fixmatch compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for semantic segmentation.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for class imbalance. Default: 1
        gamma: Focusing parameter for hard examples. Default: 2.0
        ignore_index: Target value to ignore. Default: 255
        reduction: Reduction method ('mean', 'sum', 'none'). Default: 'mean'
    """

    def __init__(self, alpha=1, gamma=2.0, ignore_index=255, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs: Predictions of shape (N, C, H, W)
            targets: Ground truth of shape (N, H, W)

        Returns:
            Focal loss (scalar if reduction='mean'/'sum', per-pixel if 'none')
        """
        # Compute cross entropy loss per pixel (no reduction)
        ce_loss = F.cross_entropy(
            inputs, targets, ignore_index=self.ignore_index, reduction="none"
        )

        # Compute pt (probability of correct class)
        pt = torch.exp(-ce_loss)

        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            # Mask out ignore_index pixels for mean calculation
            valid_mask = targets != self.ignore_index
            if valid_mask.sum() > 0:
                return focal_loss[valid_mask].mean()
            else:
                return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
