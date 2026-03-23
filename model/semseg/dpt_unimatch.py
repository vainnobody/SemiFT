"""
DPT_UniMatch: DPT model with Feature Perturbation support for UniMatch training.

This model extends the standard DPT architecture with a `need_fp` parameter
that enables feature perturbation (Dropout2d on features) similar to
UniMatch's DeepLabV3Plus implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3
from model.backbone.resnet import ResNet101Backbone
from model.util.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        feature_kind="token",
    ):
        super(DPTHead, self).__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels] * len(out_channels)

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for in_ch, out_channel in zip(in_channels, out_channels)
            ]
        )

        if feature_kind == "token":
            self.resize_layers = nn.ModuleList(
                [
                    nn.ConvTranspose2d(
                        in_channels=out_channels[0],
                        out_channels=out_channels[0],
                        kernel_size=4,
                        stride=4,
                        padding=0,
                    ),
                    nn.ConvTranspose2d(
                        in_channels=out_channels[1],
                        out_channels=out_channels[1],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                    ),
                    nn.Identity(),
                    nn.Conv2d(
                        in_channels=out_channels[3],
                        out_channels=out_channels[3],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ]
            )
        else:
            self.resize_layers = nn.ModuleList([nn.Identity() for _ in out_channels])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if x.dim() == 3:
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPT_UniMatch(nn.Module):
    """
    DPT model with UniMatch-style Feature Perturbation support.

    When need_fp=True, returns (out, out_fp) where out_fp is computed
    from Dropout2d-perturbed features.
    """

    def __init__(
        self,
        encoder_size="base",
        nclass=21,
        features=128,
        out_channels=[96, 192, 384, 768],
        use_bn=False,
        backbone_version="dinov2",
    ):
        super(DPT_UniMatch, self).__init__()

        # Intermediate layer indices for feature extraction
        self.intermediate_layer_idx_v2 = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }

        self.intermediate_layer_idx_v3 = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [5, 11, 17, 23],
            "so400m": [6, 13, 20, 26],
            "huge": [7, 15, 23, 31],
            "giant": [9, 19, 29, 39],
        }

        self.encoder_size = encoder_size
        self.backbone_version = backbone_version

        if backbone_version == "dinov2":
            self.backbone = DINOv2(model_name=encoder_size)
            self.intermediate_layer_idx = self.intermediate_layer_idx_v2
        elif backbone_version == "dinov3":
            self.backbone = DINOv3(model_name=encoder_size)
            self.intermediate_layer_idx = self.intermediate_layer_idx_v3
        elif backbone_version == "resnet":
            self.backbone = ResNet101Backbone()
            self.intermediate_layer_idx = None
        else:
            raise ValueError(
                f"Unknown backbone version: {backbone_version}. Use 'dinov2', 'dinov3', or 'resnet'."
            )
        self.feature_kind = getattr(self.backbone, "feature_kind", "token")

        self.head = DPTHead(
            nclass,
            self.backbone.out_channels if self.feature_kind == "feature_map" else self.backbone.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            feature_kind=self.feature_kind,
        )

        # Feature perturbation dropout
        self.fp_dropout = nn.Dropout2d(0.5)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x, need_fp=False):
        """
        Forward pass with optional feature perturbation.

        Args:
            x: Input tensor of shape (B, C, H, W)
            need_fp: If True, returns (out, out_fp) where out_fp is computed
                    from Dropout2d-perturbed features

        Returns:
            If need_fp=False: out tensor of shape (B, nclass, H, W)
            If need_fp=True: tuple (out, out_fp) each of shape (B, nclass, H, W)
        """
        if self.feature_kind == "feature_map":
            patch_h = patch_w = None
            features = self.backbone.forward_features(x)
        else:
            patch_size = self.backbone.patch_size
            patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
            features = self.backbone.get_intermediate_layers(
                x, self.intermediate_layer_idx[self.encoder_size]
            )
        # features is a tuple of tensors, each of shape (B, num_patches, embed_dim)

        if need_fp:
            # Feature perturbation mode: apply Dropout2d to features
            # Need to reshape to apply 2D dropout, then reshape back
            features_fp = []
            for feat in features:
                if feat.dim() == 4:
                    B, D, _, _ = feat.shape
                    feat_2d = feat
                else:
                    B, N, D = feat.shape
                    feat_2d = feat.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)
                # Apply dropout
                feat_2d_fp = self.fp_dropout(feat_2d)
                if feat.dim() == 4:
                    feat_fp = feat_2d_fp
                else:
                    feat_fp = feat_2d_fp.reshape(B, D, N).permute(0, 2, 1)
                features_fp.append(feat_fp)

            # Concatenate normal and perturbed features along batch dimension
            features_combined = [
                torch.cat([f, f_fp], dim=0) for f, f_fp in zip(features, features_fp)
            ]

            # Forward through head
            out = self.head(features_combined, patch_h, patch_w)
            out = F.interpolate(
                out,
                x.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

            # Split back to normal and perturbed outputs
            out, out_fp = out.chunk(2, dim=0)
            return out, out_fp

        # Normal mode
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(
            out,
            x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        return out
