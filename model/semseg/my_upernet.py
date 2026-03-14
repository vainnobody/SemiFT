"""
MyUperNet: mmseg-style UperNet with DINOv2/DINOv3 backbone and DPT-compatible interface.
Combines the powerful UPerHead decoder from mmseg with DINO backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3
from model.semseg.feature_perturb import apply_structured_feature_perturbation
from model.semseg.encoder_decoder import MTP_SS_UperNet


class MyUperNet(nn.Module):
    """
    UperNet with mmseg decoder and DINOv2/DINOv3 backbone.

    Uses mmseg's UPerHead for decoding, compatible with DPT interface in SemiFT.
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
        channels=256,  # UPerHead output channels
        use_bn=True,  # kept for API compatibility with DPT
        backbone_version="dinov2",
        **kwargs,  # Ignore DPT-specific params (features, out_channels)
    ):
        super(MyUperNet, self).__init__()

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
        else:
            raise ValueError(
                f"Unknown backbone version: {backbone_version}. Use 'dinov2' or 'dinov3'."
            )

        embed_dim = self.backbone.embed_dim

        # mmseg-style UperNet decoder
        # in_channels: 4 levels with same embed_dim (DINO outputs same dim for all layers)
        in_channels = [embed_dim] * 4
        self.decoder = MTP_SS_UperNet(
            decode_head=dict(
                type="UPerHead",
                num_classes=1,  # Not used, we use separate semseghead
                in_channels=in_channels,
                ignore_index=255,
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=channels,
                dropout_ratio=0.1,
                norm_cfg=dict(type="SyncBN", requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
            )
        )

        # Final classification head
        self.semseghead = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(channels, nclass, kernel_size=1)
        )

        # For comp_drop support (same as DPT)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    @property
    def head(self):
        """Return decoder components for compatibility with DPT interface."""
        return nn.ModuleList([self.decoder, self.semseghead])

    def lock_backbone(self):
        """Lock backbone parameters (same interface as DPT)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x, comp_drop=False, feature_perturb=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)
            comp_drop: Whether to apply complementary dropout (same as DPT)
            feature_perturb: Optional structured perturbation config

        Returns:
            Segmentation logits of shape (B, nclass, H, W)
        """
        patch_size = self.backbone.patch_size
        B, C, H, W = x.shape
        patch_h, patch_w = H // patch_size, W // patch_size

        # Get intermediate layer features
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )

        # Apply complementary dropout if enabled
        if comp_drop:
            bs, dim = features[0].shape[0], features[0].shape[-1]

            dropout_mask1 = self.binomial.sample((bs // 2, dim)).cuda() * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2)[:num_kept]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0

            dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
            features = tuple(
                feature * dropout_mask.unsqueeze(1).to(feature.device)
                for feature in features
            )

        # Reshape features from (B, N, C) to (B, C, H, W)
        feat_maps = []
        for feat in features:
            feat = feat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
            feat = feat.float()
            if feature_perturb is not None:
                feat = apply_structured_feature_perturbation(feat, feature_perturb)
            feat_maps.append(feat)

        # Decoder: use mmseg's UPerHead _forward_feature
        ss = self.decoder.decode_head._forward_feature(feat_maps)

        # Classification head
        out = self.semseghead(ss)

        # Upsample to original resolution
        out = F.interpolate(
            out,
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )

        return out
