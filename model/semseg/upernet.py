"""
UperNet implementation for SemiFT project.
Migrated from dinov3_segmentation/upernet.py with DPT-style interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3
from model.backbone.resnet import ResNet101Backbone
from model.semseg.feature_perturb import apply_structured_feature_perturbation


class PPM(nn.Module):
    """Pyramid Pooling Module - mimics mmsegmentation implementation."""

    def __init__(
        self, in_channels, out_channels, pool_scales=(1, 2, 3, 6), dropout=0.1
    ):
        super().__init__()
        self.pool_scales = pool_scales
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for scale in pool_scales
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        ppm_outs = [x] + [
            F.interpolate(
                stage(x), size=x.shape[2:], mode="bilinear", align_corners=False
            )
            for stage in self.stages
        ]
        concat = torch.cat(ppm_outs, dim=1)
        return self.bottleneck(concat)


class Feature2Pyramid(nn.Module):
    """Convert features to multi-scale pyramid."""

    def __init__(self, embed_dim, rescales=[4, 2, 1, 0.5]):
        super().__init__()
        self.rescales = rescales
        self.ops = nn.ModuleList()
        for r in rescales:
            if r == 4:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim, embed_dim, kernel_size=2, stride=2, bias=False
                        ),
                        nn.BatchNorm2d(embed_dim),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            embed_dim, embed_dim, kernel_size=2, stride=2
                        ),
                    )
                )
            elif r == 2:
                self.ops.append(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
                )
            elif r == 1:
                self.ops.append(nn.Identity())
            elif r == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise KeyError(f"Invalid rescale factor: {r}")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outs = []
        for i, feat in enumerate(inputs):
            x = self.ops[i](feat)
            outs.append(x)
        return tuple(outs)


class UPerNetDecoder(nn.Module):
    """UPerNet decoder without auxiliary head."""

    def __init__(
        self,
        in_channels,
        ppm_channels=512,
        fpn_channels=512,
        num_classes=21,
        dropout=0.1,
    ):
        super().__init__()

        # Lateral convolutions for FPN
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, fpn_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=False),
                )
                for ch in in_channels[:-1]
            ]
        )

        # FPN convolutions
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=False),
                )
                for _ in in_channels[:-1]
            ]
        )

        # PPM for the last feature map
        self.ppm = PPM(
            in_channels[-1], fpn_channels, pool_scales=(1, 2, 3, 6), dropout=dropout
        )

        # FPN bottleneck
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                len(in_channels) * fpn_channels, fpn_channels, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=dropout),
        )

        # Final classifier
        self.classifier = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)

    def forward(self, feats, return_feats=False):
        assert len(feats) == 4, "Expecting [P2, P3, P4, P5] features"

        # Lateral + top-down FPN
        laterals = [l_conv(feats[i]) for i, l_conv in enumerate(self.lateral_convs)]
        top = self.ppm(feats[-1])
        laterals.append(top)

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + up

        # FPN outputs
        fpn_outs = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]
        fpn_outs.append(laterals[-1])  # PPM output

        # Upsample all to the same size
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Concatenate and classify
        concat = torch.cat(fpn_outs, dim=1)
        out = self.fpn_bottleneck(concat)
        logits = self.classifier(out)
        if return_feats:
            return logits, out
        return logits


class UperNet(nn.Module):
    """
    UperNet for semantic segmentation with DINOv2/DINOv3 backbone.

    Interface compatible with DPT model in SemiFT.
    """

    # Embedding dimensions for different model sizes
    DIM_SIZE = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    def __init__(
        self,
        encoder_size="base",
        nclass=21,
        fpn_channels=256,
        use_bn=True,  # kept for API compatibility with DPT
        backbone_version="dinov2",
        **kwargs,  # Ignore DPT-specific params (features, out_channels)
    ):
        super(UperNet, self).__init__()

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

        # Initialize backbone
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

        if self.feature_kind == "token":
            embed_dim = self.backbone.embed_dim
            self.neck = Feature2Pyramid(embed_dim=embed_dim)
            in_channels = [embed_dim] * 4
        else:
            self.neck = nn.Identity()
            in_channels = list(self.backbone.out_channels)

        # UperNet decoder
        self.decoder = UPerNetDecoder(
            in_channels=in_channels,
            fpn_channels=fpn_channels,
            num_classes=nclass,
        )

        # For comp_drop support (same as DPT)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)
        self.fp_dropout = nn.Dropout2d(0.5)

    @property
    def head(self):
        """Return decoder components for compatibility with DPT interface."""
        return nn.ModuleList([self.neck, self.decoder])

    def lock_backbone(self):
        """Lock backbone parameters (same interface as DPT)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x, comp_drop=False, feature_perturb=None, need_fp=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)
            comp_drop: Whether to apply complementary dropout (same as DPT)
            feature_perturb: Optional structured perturbation config
            need_fp: Whether to return an official-UniMatch style feature-perturbed
                auxiliary prediction branch.

        Returns:
            If need_fp=False: segmentation logits of shape (B, nclass, H, W)
            If need_fp=True: tuple (out, out_fp), each of shape (B, nclass, H, W)
        """
        if need_fp and comp_drop:
            raise ValueError("UPerNet does not support need_fp=True together with comp_drop=True.")

        B, C, H, W = x.shape
        if self.feature_kind == "token":
            patch_size = self.backbone.patch_size
            patch_h, patch_w = H // patch_size, W // patch_size
            features = self.backbone.get_intermediate_layers(
                x, self.intermediate_layer_idx[self.encoder_size]
            )
        else:
            patch_h = patch_w = None
            features = self.backbone.forward_features(x)

        if need_fp:
            feat_maps = []
            feat_maps_fp = []
            for feat in features:
                if feat.dim() == 3:
                    feat = feat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
                feat = feat.float()
                feat_fp = self.fp_dropout(feat)
                if feature_perturb is not None:
                    feat = apply_structured_feature_perturbation(feat, feature_perturb)
                    feat_fp = apply_structured_feature_perturbation(
                        feat_fp, feature_perturb
                    )
                feat_maps.append(feat)
                feat_maps_fp.append(feat_fp)

            pyramid_feats = self.neck(tuple(torch.cat((feat, feat_fp), dim=0) for feat, feat_fp in zip(feat_maps, feat_maps_fp)))
            logits = self.decoder(pyramid_feats)
            out = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            return out.chunk(2, dim=0)

        # Apply complementary dropout if enabled
        if comp_drop:
            if features[0].dim() == 4:
                bs, dim = features[0].shape[0], features[0].shape[1]
                dropout_mask1 = (
                    self.binomial.sample((bs // 2, dim)).to(features[0].device).unsqueeze(-1).unsqueeze(-1) * 2.0
                )
            else:
                bs, dim = features[0].shape[0], features[0].shape[-1]
                dropout_mask1 = self.binomial.sample((bs // 2, dim)).to(features[0].device) * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2)[:num_kept]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0

            dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
            if features[0].dim() == 4:
                features = tuple(feature * dropout_mask.to(feature.device) for feature in features)
            else:
                features = tuple(
                    feature * dropout_mask.unsqueeze(1).to(feature.device)
                    for feature in features
                )

        feat_maps = []
        for feat in features:
            if feat.dim() == 3:
                feat = feat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
            feat = feat.float()
            if feature_perturb is not None:
                feat = apply_structured_feature_perturbation(feat, feature_perturb)
            feat_maps.append(feat)

        pyramid_feats = self.neck(tuple(feat_maps))

        # Decoder: get segmentation logits
        logits = self.decoder(pyramid_feats)

        # Upsample to original resolution
        out = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return out
