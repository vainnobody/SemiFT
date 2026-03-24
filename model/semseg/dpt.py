import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3
from model.backbone.resnet import ResNet101Backbone
from model.semseg.scalematch_core import (
    RWKVLayers,
    SqueezeExcitation,
    resize_x,
    scale_as,
)
from model.semseg.feature_perturb import apply_structured_feature_perturbation
from model.semseg.corrmatch_utils import Corr
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
        self.feature_kind = feature_kind

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

    def forward(self, out_features, patch_h, patch_w, return_feats=False):
        out = []
        for i, x in enumerate(out_features):
            if x.dim() == 3:
                x = x.permute(0, 2, 1).reshape(
                    (x.shape[0], x.shape[-1], patch_h, patch_w)
                )
            elif x.dim() != 4:
                raise ValueError(f"Unsupported feature rank: {x.dim()}")

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
        if return_feats:
            return out, path_1
        return out


class DPT(nn.Module):
    def __init__(
        self,
        encoder_size="base",
        nclass=21,
        features=128,
        out_channels=[96, 192, 384, 768],
        use_bn=False,
        backbone_version="dinov2",  # 'dinov2' or 'dinov3'
        enable_scalematch=False,
        enable_corrmatch=False,
        enable_segmind=False,
        proj_dim=256,
    ):
        super(DPT, self).__init__()

        # Intermediate layer indices for feature extraction
        # DINOv2 layer indices
        self.intermediate_layer_idx_v2 = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }

        # DINOv3 layer indices (depth varies by model size)
        self.intermediate_layer_idx_v3 = {
            "small": [2, 5, 8, 11],  # depth=12
            "base": [2, 5, 8, 11],  # depth=12
            "large": [5, 11, 17, 23],  # depth=24
            "so400m": [6, 13, 20, 26],  # depth=27
            "huge": [7, 15, 23, 31],  # depth=32
            "giant": [9, 19, 29, 39],  # depth=40
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
        self.enable_scalematch = enable_scalematch
        self.enable_corrmatch = enable_corrmatch
        self.enable_segmind = enable_segmind
        self.scalematch_features = features

        self.head = DPTHead(
            nclass,
            self.backbone.out_channels if self.feature_kind == "feature_map" else self.backbone.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            feature_kind=self.feature_kind,
        )
        if self.enable_segmind:
            self.segmind_projector = nn.Sequential(
                nn.Conv2d(features, proj_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(proj_dim, proj_dim, kernel_size=1, bias=False),
            )
            recon_hidden = max(features // 2, 32)
            self.segmind_reconstruction_head = nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, recon_hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(recon_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(recon_hidden, 3, kernel_size=1),
                nn.Tanh(),
            )

        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)
        self.fp_dropout = nn.Dropout2d(0.5)
        if self.enable_corrmatch:
            self.corr_proj = nn.Sequential(
                nn.Conv2d(features, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )
            self.corr = Corr(nclass=nclass)
        if self.enable_scalematch:
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

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _extract_features(self, x):
        if self.feature_kind == "feature_map":
            return self.backbone.forward_features(x), None, None
        patch_size = self.backbone.patch_size
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )
        return features, patch_h, patch_w

    def _apply_feature_perturbation(self, features, patch_h, patch_w, batch_size, feature_perturb):
        perturbed_features = []
        for feature in features:
            if feature.dim() == 4:
                feat_map = feature
            else:
                feat_map = feature.permute(0, 2, 1).reshape(
                    batch_size, feature.shape[-1], patch_h, patch_w
                )
            feat_map = apply_structured_feature_perturbation(
                feat_map.float(), feature_perturb
            )
            if feature.dim() == 4:
                perturbed_feature = feat_map.to(dtype=feature.dtype)
            else:
                perturbed_feature = feat_map.reshape(batch_size, feature.shape[-1], -1)
                perturbed_feature = perturbed_feature.permute(0, 2, 1).to(
                    dtype=feature.dtype
                )
            perturbed_features.append(perturbed_feature)
        return tuple(perturbed_features)

    def _apply_unimatch_feature_dropout(self, features, patch_h, patch_w):
        features_fp = []
        for feature in features:
            if feature.dim() == 4:
                feat_map = feature
            else:
                batch_size, num_tokens, dim = feature.shape
                feat_map = feature.permute(0, 2, 1).reshape(
                    batch_size, dim, patch_h, patch_w
                )

            feat_map_fp = self.fp_dropout(feat_map)

            if feature.dim() == 4:
                feature_fp = feat_map_fp
            else:
                feature_fp = feat_map_fp.reshape(batch_size, dim, num_tokens).permute(
                    0, 2, 1
                )
            features_fp.append(feature_fp)

        return tuple(features_fp)

    def _scalematch_base_forward(self, x, need_fp=False, feature_scale=None):
        features, patch_h, patch_w = self._extract_features(x)
        logits, feats = self.head(features, patch_h, patch_w, return_feats=True)
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).contiguous()
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

        fp_inputs = torch.cat((feats_fp, self.fp_dropout(feats_fp)), dim=0)
        logits_fp = self.head.scratch.output_conv(fp_inputs)
        logits_fp = F.interpolate(
            logits_fp,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).contiguous()
        _, logits_fp = logits_fp.chunk(2)
        return logits, feats_fp, logits_fp

    def _scalematch_two_scale_forward(self, inputs, scale_factor, feature_scale):
        resize_stride = getattr(
            self.backbone,
            "output_stride",
            getattr(self.backbone, "patch_size", 14),
        )
        if scale_factor is None:
            logits, _ = self._scalematch_base_forward(inputs, need_fp=False)
            return logits

        x_1x = inputs
        if scale_factor > 1.0:
            x_lo = x_1x
            x_hi = resize_x(x_1x, scale_factor, patch_size=resize_stride, align_corners=True)
            p_lo_ori, feats_lo, out_fp = self._scalematch_base_forward(
                x_lo, need_fp=True, feature_scale=feature_scale
            )
            p_hi, feats_hi = self._scalematch_base_forward(x_hi)

            p_hi = scale_as(p_hi, x_1x, align_corners=True).contiguous()
            feats_hi = scale_as(feats_hi, feats_lo, align_corners=True).contiguous()
            cat_feats = torch.cat([feats_lo, feats_hi], 1).contiguous()

            global_int_feats, h_f, w_f = self.rwkv_layers(cat_feats)
            bsz, _, ch = global_int_feats.shape
            global_int_feats = global_int_feats.permute(0, 2, 1).reshape(
                bsz, ch, h_f, w_f
            ).contiguous()
            channel_attn_feats = self.se_block(
                torch.cat([cat_feats, global_int_feats], 1).contiguous()
            )
            logit_attn = scale_as(
                self.scale_attn(channel_attn_feats), p_lo_ori, align_corners=True
            ).contiguous()

            p_lo = logit_attn * p_lo_ori
            p_lo_up = scale_as(p_lo, p_hi, align_corners=True).contiguous()
            logit_attn_hi = scale_as(logit_attn, p_hi, align_corners=True).contiguous()
            joint_pred = (p_lo_up + (1 - logit_attn_hi) * p_hi).contiguous()
            joint_pred = scale_as(joint_pred, p_lo_ori, align_corners=True).contiguous()

            return {
                "pred_joint": joint_pred,
                "pred_ori": p_lo_ori,
                "pred_fp": out_fp,
                "pred_size": p_hi,
            }

        x_lo = resize_x(x_1x, scale_factor, patch_size=resize_stride, align_corners=True)
        x_hi = x_1x
        p_lo, feats_lo = self._scalematch_base_forward(x_lo)
        p_hi, feats_hi, out_fp = self._scalematch_base_forward(
            x_hi, need_fp=True, feature_scale=feature_scale
        )

        p_lo_ori = scale_as(p_lo, x_1x, align_corners=True).contiguous()
        feats_lo = scale_as(feats_lo, feats_hi, align_corners=True).contiguous()
        cat_feats = torch.cat([feats_lo, feats_hi], 1).contiguous()

        global_int_feats, h_f, w_f = self.rwkv_layers(cat_feats)
        bsz, _, ch = global_int_feats.shape
        global_int_feats = global_int_feats.permute(0, 2, 1).reshape(
            bsz, ch, h_f, w_f
        ).contiguous()
        channel_attn_feats = self.se_block(
            torch.cat([cat_feats, global_int_feats], 1).contiguous()
        )
        logit_attn = scale_as(
            self.scale_attn(channel_attn_feats), p_lo, align_corners=True
        ).contiguous()

        p_lo_att = (logit_attn * p_lo).contiguous()
        p_lo_att = scale_as(p_lo_att, p_hi, align_corners=True).contiguous()
        logit_attn_hi = scale_as(logit_attn, p_hi, align_corners=True).contiguous()
        joint_pred = (p_lo_att + (1 - logit_attn_hi) * p_hi).contiguous()

        return {
            "pred_joint": joint_pred,
            "pred_ori": p_hi,
            "pred_fp": out_fp,
            "pred_size": p_lo_ori,
        }

    def _corrmatch_forward(
        self,
        x,
        features,
        patch_h,
        patch_w,
        need_fp=False,
    ):
        logits, corr_feats = self.head(features, patch_h, patch_w, return_feats=True)
        logits = F.interpolate(
            logits,
            x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        outputs = {"out": logits}

        if need_fp:
            features_fp = self._apply_unimatch_feature_dropout(
                features, patch_h, patch_w
            )
            logits_fp = self.head(features_fp, patch_h, patch_w)
            logits_fp = F.interpolate(
                logits_fp,
                x.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            outputs["out_fp"] = logits_fp

        corr_inputs = self.corr_proj(corr_feats)
        corr_dict = self.corr(corr_inputs, logits)
        outputs["corr_map"] = corr_dict["corr_map"]
        outputs["corr_out"] = F.interpolate(
            corr_dict["out"],
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        return outputs

    def _segmind_forward(self, x, features, patch_h, patch_w):
        logits, decoder_feats = self.head(features, patch_h, patch_w, return_feats=True)
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        proj_feat = self.segmind_projector(decoder_feats)
        proj_feat = F.interpolate(
            proj_feat,
            size=(x.shape[-2] // 4, x.shape[-1] // 4),
            mode="bilinear",
            align_corners=True,
        )
        recon = self.segmind_reconstruction_head(decoder_feats)
        recon = F.interpolate(
            recon,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        return logits, proj_feat, recon

    def forward(
        self,
        x,
        comp_drop=False,
        feature_perturb=None,
        need_fp=False,
        use_corr=False,
        scale_factor=None,
        feature_scale=1.0,
        return_aux=False,
        mode=None,
        mask=None,
    ):
        if scale_factor is not None:
            if not self.enable_scalematch:
                raise ValueError("ScaleMatch forward requested but enable_scalematch=False.")
            if need_fp or feature_perturb is not None or comp_drop or use_corr or return_aux or mode is not None:
                raise ValueError("ScaleMatch forward does not support comp_drop/need_fp/feature_perturb/use_corr/return_aux/mode.")
            return self._scalematch_two_scale_forward(x, scale_factor, feature_scale)

        batch_size = x.shape[0]
        features, patch_h, patch_w = self._extract_features(x)

        if need_fp and comp_drop:
            raise ValueError("DPT does not support need_fp=True together with comp_drop=True.")
        if use_corr and not self.enable_corrmatch:
            raise ValueError("CorrMatch forward requested but enable_corrmatch=False.")
        if mode is not None and mode != "r":
            raise ValueError(f"Unsupported DPT mode: {mode}")
        if (return_aux or mode == "r") and not self.enable_segmind:
            raise ValueError("SegMind auxiliary outputs requested but enable_segmind=False.")
        if (return_aux or mode == "r") and (need_fp or comp_drop or use_corr):
            raise ValueError("SegMind auxiliary outputs do not support need_fp/comp_drop/use_corr.")

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

            out = self.head(features, patch_h, patch_w)
            return F.interpolate(
                out,
                x.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        if feature_perturb is not None:
            features = self._apply_feature_perturbation(
                features, patch_h, patch_w, batch_size, feature_perturb
            )

        if return_aux or mode == "r":
            return self._segmind_forward(x, features, patch_h, patch_w)

        if use_corr:
            return self._corrmatch_forward(
                x,
                features,
                patch_h,
                patch_w,
                need_fp=need_fp,
            )

        if need_fp:
            features_fp = self._apply_unimatch_feature_dropout(
                features, patch_h, patch_w
            )
            features = tuple(
                torch.cat((feature, feature_fp), dim=0)
                for feature, feature_fp in zip(features, features_fp)
            )
            out = self.head(features, patch_h, patch_w)
            out = F.interpolate(
                out,
                x.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            out, out_fp = out.chunk(2, dim=0)
            return out, out_fp

        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, x.shape[-2:], mode="bilinear", align_corners=True)

        return out
