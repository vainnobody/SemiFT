"""UPerNet decoder with ScaleMatch multi-scale training support."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3
from model.semseg.dpt_scalematch import (
    RWKVLayers,
    SqueezeExcitation,
    resize_x,
    scale_as,
)
from model.semseg.upernet import Feature2Pyramid, UPerNetDecoder


class UperNet_ScaleMatch(nn.Module):
    def __init__(
        self,
        encoder_size="base",
        nclass=21,
        fpn_channels=256,
        backbone_version="dinov2",
        decoder_dropout=0.1,
        fp_dropout=0.5,
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

        embed_dim = self.backbone.embed_dim
        self.neck = Feature2Pyramid(embed_dim=embed_dim)
        self.decoder = UPerNetDecoder(
            in_channels=[embed_dim] * 4,
            fpn_channels=fpn_channels,
            num_classes=nclass,
            dropout=decoder_dropout,
        )
        self.head = nn.ModuleList([self.neck, self.decoder])

        scale_in_ch = 2 * fpn_channels
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
        self.feature_dropout = nn.Dropout2d(fp_dropout)

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

    def _reshape_tokens(self, token_features, patch_h, patch_w):
        feat_maps = []
        for feat in token_features:
            feat_maps.append(
                feat.permute(0, 2, 1)
                .reshape(feat.shape[0], feat.shape[-1], patch_h, patch_w)
                .contiguous()
                .float()
            )
        return tuple(feat_maps)

    def _base_forward(self, x, need_fp=False, feature_scale=None):
        token_features, patch_h, patch_w = self._extract_features(x)
        pyramid_feats = self.neck(self._reshape_tokens(token_features, patch_h, patch_w))
        logits, feats = self.decoder(pyramid_feats, return_feats=True)
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
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
                align_corners=False,
            ).contiguous()

        fp_inputs = torch.cat((feats_fp, self.feature_dropout(feats_fp)), dim=0)
        fp_logits = self.decoder.classifier(fp_inputs)
        fp_logits = F.interpolate(
            fp_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        ).contiguous()
        _, out_fp = fp_logits.chunk(2)
        return logits, feats_fp, out_fp

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
            global_int_feats = (
                global_int_feats.permute(0, 2, 1).reshape(bsz, ch, h_f, w_f).contiguous()
            )
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
        global_int_feats = (
            global_int_feats.permute(0, 2, 1).reshape(bsz, ch, h_f, w_f).contiguous()
        )
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
    ):
        del scales, comp_drop
        return self.two_scale_forward(x, scale_factor, feature_scale)
