"""
DPT with ScaleMatch multi-scale training support.

Based on DPT architecture from dpt.py with added multi-scale forward pass
and scale attention mechanisms from ScaleMatch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath

from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3
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


def scale_as(x, target):
    """Scale x to the same size as target tensor."""
    if x.shape[-2:] == target.shape[-2:]:
        return x
    return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=True)


def resize_x(x, scale_factor, patch_size=14):
    """Resize x by a scale factor, ensuring patch alignment."""
    if scale_factor == 1.0:
        return x

    H, W = x.shape[-2:]
    target_H = int(H * scale_factor)
    target_W = int(W * scale_factor)

    # Snap to nearest multiple of patch_size
    target_H = round(target_H / patch_size) * patch_size
    target_W = round(target_W / patch_size) * patch_size

    # Ensure at least one patch
    target_H = max(target_H, patch_size)
    target_W = max(target_W, patch_size)

    return F.interpolate(
        x, size=(target_H, target_W), mode="bilinear", align_corners=True
    )


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation attention block."""

    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class RWKVBlock(nn.Module):
    """Simplified RWKV-style block for global feature interaction."""

    def __init__(
        self, channels, mlp_ratio=4.0, drop_path=0.0, layer_id=0, total_layers=2
    ):
        super().__init__()
        self.layer_id = layer_id
        hidden_dim = int(channels * mlp_ratio)

        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)

        # Spatial mixing (simplified attention)
        self.spatial_mix = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        # Channel mixing (MLP)
        self.channel_mix = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer scale
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True
        )

    def forward(self, x, H, W):
        # x: (B, N, C) where N = H * W
        x = x + self.drop_path(self.layer_scale_1 * self.spatial_mix(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.channel_mix(self.norm2(x)))
        return x


class RWKVLayers(nn.Module):
    """Stack of RWKV blocks for global feature interaction."""

    def __init__(
        self, num_layers, channels, mlp_ratio=4.0, drop_path=0.0, total_layers=2
    ):
        super().__init__()

        # Reduce concatenated features to channels
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 16, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.blocks = nn.ModuleList(
            [
                RWKVBlock(
                    channels=channels,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    total_layers=total_layers,
                    layer_id=i,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x, H, W):
        # x: (B, C, H, W)
        x = self.reduce(x)
        B, C, _, _ = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        for blk in self.blocks:
            x = blk(x, H, W)

        x = self.norm(x)
        return x


class DPTHead(nn.Module):
    """DPT decoder head."""

    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

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

        self.features = features

    def forward(self, out_features, patch_h, patch_w, return_feats=False):
        out = []
        for i, x in enumerate(out_features):
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

        logits = self.scratch.output_conv(path_1)

        if return_feats:
            return logits, path_1
        return logits


class DPT_ScaleMatch(nn.Module):
    """
    DPT with ScaleMatch multi-scale training.

    Combines DPT decoder with multi-scale forward pass and scale attention
    mechanisms from ScaleMatch for improved semi-supervised segmentation.
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
        super(DPT_ScaleMatch, self).__init__()

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

        # DPT decoder head
        self.head = DPTHead(
            nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels
        )

        # ScaleMatch specific components
        scale_in_ch = 2 * features  # Concatenated features from two scales

        # Scale attention module
        self.scale_attn = nn.Sequential(
            nn.Conv2d(
                scale_in_ch + 32,
                scale_in_ch + 32,
                kernel_size=3,
                padding=1,
                groups=scale_in_ch + 32,
                bias=False,
            ),
            nn.BatchNorm2d(scale_in_ch + 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale_in_ch + 32, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Squeeze-and-Excitation block
        self.se_block = SqueezeExcitation(scale_in_ch + 32)

        # Global interaction via RWKV-style layers
        self.rwkv_layers = RWKVLayers(
            1, scale_in_ch // 16, mlp_ratio=4.0, drop_path=0.0, total_layers=2
        )

        # Dropout for feature perturbation
        self.feature_dropout = nn.Dropout2d(0.1)

        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)
        self.features = features

    @property
    def lock_backbone(self):
        """Lock backbone parameters."""

        def _lock():
            for p in self.backbone.parameters():
                p.requires_grad = False

        return _lock

    def _extract_features(self, x):
        """Extract intermediate features from backbone."""
        patch_size = self.backbone.patch_size
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size

        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )
        return features, patch_h, patch_w

    def _base_forward(self, x, need_fp=False, feature_scale=None):
        """Single-scale forward pass."""
        features, patch_h, patch_w = self._extract_features(x)

        logits, feats = self.head(features, patch_h, patch_w, return_feats=True)

        # Resize to input size
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=True
        )
        feats = F.interpolate(
            feats,
            size=(x.shape[-2] // 4, x.shape[-1] // 4),
            mode="bilinear",
            align_corners=True,
        )

        if need_fp:
            # Feature perturbation output
            feats_fp = self.feature_dropout(feats)
            logits_fp = F.interpolate(
                self.head.scratch.output_conv(feats_fp),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            return logits, feats, logits_fp

        return logits, feats

    def two_scale_forward(self, inputs, scale_factor, feature_scale):
        """
        Multi-scale forward pass from ScaleMatch.

        Args:
            inputs: Input tensor (B, 3, H, W)
            scale_factor: Scale factor for the second scale
            feature_scale: Feature scale factor (unused in current implementation)

        Returns:
            dict with keys: pred_joint, pred_ori, pred_fp, pred_size
        """
        x_1x = inputs
        B, C, H, W = x_1x.shape

        if scale_factor is None:
            out, _ = self._base_forward(x_1x)
            return out

        if scale_factor > 1.0:
            # High resolution path
            x_lo = x_1x
            x_hi = resize_x(x_1x, scale_factor, patch_size=self.backbone.patch_size)

            p_lo_ori, feats_lo, out_fp = self._base_forward(
                x_lo, need_fp=True, feature_scale=feature_scale
            )
            p_hi, feats_hi = self._base_forward(x_hi)

            p_hi = scale_as(p_hi, x_1x)
            feats_hi = scale_as(feats_hi, feats_lo)
            cat_feats = torch.cat([feats_lo, feats_hi], 1)  # (B, 2*features, H', W')

            H_f, W_f = cat_feats.size(2), cat_feats.size(3)

            # Global interaction
            global_int_feats = self.rwkv_layers(cat_feats, H_f, W_f)
            global_int_feats = rearrange(
                global_int_feats, "b (h w) c -> b c h w", h=H_f, w=W_f
            ).contiguous()

            # Channel attention
            channel_attn_feats = self.se_block(
                torch.cat([cat_feats, global_int_feats], 1)
            )

            # Scale attention
            logit_attn = self.scale_attn(channel_attn_feats)
            logit_attn = scale_as(logit_attn, p_lo_ori)

            p_lo = logit_attn * p_lo_ori
            p_lo_up = scale_as(p_lo, p_hi)
            logit_attn = scale_as(logit_attn, p_hi)
            joint_pred = p_lo_up + (1 - logit_attn) * p_hi
            joint_pred = scale_as(joint_pred, p_lo_ori)

            return {
                "pred_joint": joint_pred,
                "pred_ori": p_lo_ori,
                "pred_fp": out_fp,
                "pred_size": p_hi,
            }

        else:
            # Low resolution path
            x_lo = resize_x(x_1x, scale_factor, patch_size=self.backbone.patch_size)
            x_hi = x_1x

            p_lo, feats_lo = self._base_forward(x_lo)
            p_hi, feats_hi, out_fp = self._base_forward(
                x_hi, need_fp=True, feature_scale=feature_scale
            )

            p_lo_ori = scale_as(p_lo, x_1x)
            feats_lo = scale_as(feats_lo, feats_hi)
            cat_feats = torch.cat([feats_lo, feats_hi], 1)

            H_f, W_f = cat_feats.size(2), cat_feats.size(3)

            # Global interaction
            global_int_feats = self.rwkv_layers(cat_feats, H_f, W_f)
            global_int_feats = rearrange(
                global_int_feats, "b (h w) c -> b c h w", h=H_f, w=W_f
            ).contiguous()

            # Channel attention
            channel_attn_feats = self.se_block(
                torch.cat([cat_feats, global_int_feats], 1)
            )

            # Scale attention
            logit_attn = self.scale_attn(channel_attn_feats)
            logit_attn = scale_as(logit_attn, p_lo)

            p_lo_att = logit_attn * p_lo
            p_lo_att = scale_as(p_lo_att, p_hi)
            logit_attn = scale_as(logit_attn, p_hi)
            joint_pred = p_lo_att + (1 - logit_attn) * p_hi

            return {
                "pred_joint": joint_pred,
                "pred_ori": p_hi,
                "pred_fp": out_fp,
                "pred_size": p_lo_ori,
            }

    def forward(
        self, x, scale_factor=None, feature_scale=1.0, scales=None, comp_drop=False
    ):
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)
            scale_factor: Scale factor for multi-scale forward. If None, single-scale.
            feature_scale: Feature scale factor for feature perturbation
            scales: Not used (kept for API compatibility)
            comp_drop: Whether to apply complementary dropout

        Returns:
            - If scale_factor is None: tensor (B, nclass, H, W)
            - If scale_factor is not None: dict with pred_joint, pred_ori, pred_fp, pred_size
        """
        if comp_drop:
            # Complementary dropout mode (from original DPT)
            features, patch_h, patch_w = self._extract_features(x)
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
                feature * dropout_mask.unsqueeze(1) for feature in features
            )

            out = self.head(features, patch_h, patch_w)
            out = F.interpolate(
                out, size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            return out

        return self.two_scale_forward(x, scale_factor, feature_scale)
