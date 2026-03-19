"""DPT model with ScaleMatch multi-scale training support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3
from model.backbone.dinov2_layers.drop_path import DropPath
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
    if x.shape[-2:] == target.shape[-2:]:
        return x
    return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=True)


def resize_x(x, scale_factor, patch_size=14):
    if scale_factor == 1.0:
        return x

    h, w = x.shape[-2:]
    target_h = max(round((h * scale_factor) / patch_size) * patch_size, patch_size)
    target_w = max(round((w * scale_factor) / patch_size) * patch_size, patch_size)
    return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=True)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        hidden = max(in_channels // reduction_ratio, 1)
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class RWKVBlock(nn.Module):
    def __init__(self, channels, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        hidden_dim = int(channels * mlp_ratio)
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.spatial_mix = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.channel_mix = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(channels))
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(channels))

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.spatial_mix(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.channel_mix(self.norm2(x)))
        return x


class RWKVLayers(nn.Module):
    def __init__(self, num_layers, channels, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 16, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.blocks = nn.ModuleList(
            [RWKVBlock(channels, mlp_ratio=mlp_ratio, drop_path=dpr[i]) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        x = self.reduce(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
    ):
        super().__init__()
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1, padding=0)
                for out_channel in out_channels
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
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
    def __init__(
        self,
        encoder_size="base",
        nclass=21,
        features=128,
        out_channels=[96, 192, 384, 768],
        use_bn=False,
        backbone_version="dinov2",
    ):
        super().__init__()
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
        else:
            raise ValueError(
                f"Unknown backbone version: {backbone_version}. Use 'dinov2' or 'dinov3'."
            )

        self.head = DPTHead(
            nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels
        )

        scale_in_ch = 2 * features
        rwkv_channels = max(scale_in_ch // 16, 1)
        self.scale_attn = nn.Sequential(
            nn.Conv2d(
                scale_in_ch + rwkv_channels,
                scale_in_ch + rwkv_channels,
                kernel_size=3,
                padding=1,
                groups=scale_in_ch + rwkv_channels,
                bias=False,
            ),
            nn.BatchNorm2d(scale_in_ch + rwkv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale_in_ch + rwkv_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.se_block = SqueezeExcitation(scale_in_ch + rwkv_channels)
        self.rwkv_layers = RWKVLayers(1, rwkv_channels, mlp_ratio=4.0, drop_path=0.0)
        self.feature_dropout = nn.Dropout2d(0.5)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _extract_features(self, x):
        patch_size = self.backbone.patch_size
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )
        return features, patch_h, patch_w

    def _base_forward(self, x, need_fp=False, feature_scale=None):
        features, patch_h, patch_w = self._extract_features(x)
        logits, feats = self.head(features, patch_h, patch_w, return_feats=True)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True).contiguous()
        feats = F.interpolate(
            feats,
            size=(x.shape[-2] // 4, x.shape[-1] // 4),
            mode="bilinear",
            align_corners=True,
        ).contiguous()

        if not need_fp:
            return logits, feats

        feats_fp = feats
        if feature_scale is not None and feature_scale != 1.0:
            target_h = max(int(round(feats_fp.shape[-2] * feature_scale)), 1)
            target_w = max(int(round(feats_fp.shape[-1] * feature_scale)), 1)
            feats_fp = F.interpolate(
                feats_fp,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=True,
            ).contiguous()

        fp_inputs = torch.cat((feats_fp, self.feature_dropout(feats_fp)), dim=0)
        logits_fp = self.head.scratch.output_conv(fp_inputs)
        logits_fp = F.interpolate(
            logits_fp,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).contiguous()
        _, logits_fp = logits_fp.chunk(2)
        return logits, feats_fp, logits_fp

    def _strong_forward(self, x):
        logits, _ = self._base_forward(x, need_fp=False)
        return logits.contiguous()

    def two_scale_forward(self, inputs, scale_factor, feature_scale):
        if scale_factor is None:
            base_forward_out = self._base_forward(
                inputs, need_fp=self.training, feature_scale=feature_scale
            )
            if self.training:
                out, _, _ = base_forward_out
                return out.contiguous()

            out, _ = base_forward_out
            return out

        x_1x = inputs
        if scale_factor > 1.0:
            x_lo = x_1x
            x_hi = resize_x(x_1x, scale_factor, patch_size=self.backbone.patch_size)
            p_lo_ori, feats_lo, out_fp = self._base_forward(
                x_lo, need_fp=True, feature_scale=feature_scale
            )
            p_hi, feats_hi = self._base_forward(x_hi)

            p_hi = scale_as(p_hi, x_1x).contiguous()
            feats_hi = scale_as(feats_hi, feats_lo).contiguous()
            cat_feats = torch.cat([feats_lo, feats_hi], 1).contiguous()
            h_f, w_f = cat_feats.size(2), cat_feats.size(3)

            global_int_feats = self.rwkv_layers(cat_feats)
            bsz, _, ch = global_int_feats.shape
            global_int_feats = global_int_feats.permute(0, 2, 1).reshape(bsz, ch, h_f, w_f).contiguous()
            channel_attn_feats = self.se_block(
                torch.cat([cat_feats, global_int_feats], 1).contiguous()
            )
            logit_attn = scale_as(self.scale_attn(channel_attn_feats), p_lo_ori).contiguous()

            p_lo = logit_attn * p_lo_ori
            p_lo_up = scale_as(p_lo, p_hi).contiguous()
            logit_attn_hi = scale_as(logit_attn, p_hi).contiguous()
            joint_pred = (p_lo_up + (1 - logit_attn_hi) * p_hi).contiguous()
            joint_pred = scale_as(joint_pred, p_lo_ori).contiguous()

            return {
                "pred_joint": joint_pred,
                "pred_ori": p_lo_ori,
                "pred_fp": out_fp,
                "pred_size": p_hi,
            }

        x_lo = resize_x(x_1x, scale_factor, patch_size=self.backbone.patch_size)
        x_hi = x_1x
        p_lo, feats_lo = self._base_forward(x_lo)
        p_hi, feats_hi, out_fp = self._base_forward(
            x_hi, need_fp=True, feature_scale=feature_scale
        )

        p_lo_ori = scale_as(p_lo, x_1x).contiguous()
        feats_lo = scale_as(feats_lo, feats_hi).contiguous()
        cat_feats = torch.cat([feats_lo, feats_hi], 1).contiguous()
        h_f, w_f = cat_feats.size(2), cat_feats.size(3)

        global_int_feats = self.rwkv_layers(cat_feats)
        bsz, _, ch = global_int_feats.shape
        global_int_feats = global_int_feats.permute(0, 2, 1).reshape(bsz, ch, h_f, w_f).contiguous()
        channel_attn_feats = self.se_block(torch.cat([cat_feats, global_int_feats], 1).contiguous())
        logit_attn = scale_as(self.scale_attn(channel_attn_feats), p_lo).contiguous()

        p_lo_att = (logit_attn * p_lo).contiguous()
        p_lo_att = scale_as(p_lo_att, p_hi).contiguous()
        logit_attn_hi = scale_as(logit_attn, p_hi).contiguous()
        joint_pred = (p_lo_att + (1 - logit_attn_hi) * p_hi).contiguous()

        return {
            "pred_joint": joint_pred,
            "pred_ori": p_hi,
            "pred_fp": out_fp,
            "pred_size": p_lo_ori,
        }

    def forward(
        self,
        x,
        scale_factor=None,
        feature_scale=1.0,
        scales=None,
        comp_drop=False,
        strong_inputs=None,
    ):
        del scales
        if comp_drop:
            features, patch_h, patch_w = self._extract_features(x)
            bs, dim = features[0].shape[0], features[0].shape[-1]
            dropout_mask1 = self.binomial.sample((bs // 2, dim)).to(x.device) * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2, device=x.device)[:num_kept]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0
            dropout_mask = torch.cat((dropout_mask1, dropout_mask2), dim=0)
            features = tuple(
                feature * dropout_mask.unsqueeze(1).to(feature.device) for feature in features
            )
            out = self.head(features, patch_h, patch_w)
            return F.interpolate(
                out, size=x.shape[-2:], mode="bilinear", align_corners=True
            ).contiguous()

        if strong_inputs is not None:
            return {
                "multi_scale": self.two_scale_forward(x, scale_factor, feature_scale),
                "pred_strong": self._strong_forward(strong_inputs),
            }

        return self.two_scale_forward(x, scale_factor, feature_scale)
