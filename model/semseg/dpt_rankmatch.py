"""
DPT model extended for RankMatch.
Returns intermediate features for rank-based consistency loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.semseg.dpt import DPT


class DPT_RankMatch(DPT):
    """
    DPT model extended for RankMatch semi-supervised learning.

    Key modifications:
    - Returns intermediate features for corr_loss computation
    - Supports feature perturbation (need_fp) for diversity loss
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fp_dropout = nn.Dropout2d(0.5)

    def forward(self, x, need_fp=False):
        """
        Forward pass with optional feature perturbation.

        Args:
            x: Input tensor [B, 3, H, W]
            need_fp: Whether to return feature-perturbed prediction

        Returns:
            If need_fp=False:
                out: Prediction [B, nclass, H, W]
                feat: Intermediate feature [B, C, h, w] (before final upsampling)
            If need_fp=True:
                out: Prediction [B, nclass, H, W]
                out_fp: Feature-perturbed prediction [B, nclass, H, W]
                feat: Intermediate feature [B, C, h, w]
        """
        patch_size = self.backbone.patch_size
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size

        intermediate_layers = self.intermediate_layer_idx[self.encoder_size]
        features = self.backbone.get_intermediate_layers(x, intermediate_layers)

        if need_fp:
            # Feature Perturbation: Dropout and Concatenate along batch dimension
            features_expanded = []
            for f in features:
                # f is [B, N, C], Dropout2d expects [B, C, H, W]
                B, N, C = f.shape
                f_4d = f.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
                f_drop_4d = self.fp_dropout(f_4d)
                f_drop = f_drop_4d.reshape(B, C, N).permute(0, 2, 1).contiguous()
                f_cat = torch.cat((f, f_drop), dim=0)  # [2B, N, C]
                features_expanded.append(f_cat)

            # Forward head with expanded batch
            out_expanded = self.head(features_expanded, patch_h, patch_w)
            out_expanded = F.interpolate(
                out_expanded,
                (patch_h * patch_size, patch_w * patch_size),
                mode="bilinear",
                align_corners=True,
            )

            out, out_fp = out_expanded.chunk(2, dim=0)

            # Get features for corr_loss (from the deepest backbone feature)
            feat_deepest = features[-1]  # [B, N, C]
            feat = (
                feat_deepest.permute(0, 2, 1)
                .reshape(
                    feat_deepest.shape[0], feat_deepest.shape[-1], patch_h, patch_w
                )
                .contiguous()
            )

            return out, out_fp, feat

        else:
            out = self.head(features, patch_h, patch_w)
            out = F.interpolate(
                out,
                (patch_h * patch_size, patch_w * patch_size),
                mode="bilinear",
                align_corners=True,
            )

            # Get features for corr_loss (from the deepest backbone feature)
            feat_deepest = features[-1]  # [B, N, C]
            feat = (
                feat_deepest.permute(0, 2, 1)
                .reshape(
                    feat_deepest.shape[0], feat_deepest.shape[-1], patch_h, patch_w
                )
                .contiguous()
            )

            return out, feat
