import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.semseg.dpt import DPT
from model.semseg.upernet import UperNet
from model.semseg.feature_perturb import apply_structured_feature_perturbation


class CorrPropagationHead(nn.Module):
    def __init__(self, in_channels, nclass, proj_channels=256, sample_rows=128):
        super().__init__()
        self.sample_rows = sample_rows
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.conv1 = nn.Conv2d(proj_channels, nclass, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(proj_channels, nclass, kernel_size=1, bias=True)

    def _sample_corr_rows(self, corr_map):
        batch, hw, _ = corr_map.shape
        rows = min(self.sample_rows, hw)
        if rows == hw:
            return corr_map
        index = torch.randperm(hw, device=corr_map.device)[:rows]
        return corr_map[:, index, :]

    def _normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
        batch, rows, hw = corr_map.shape
        corr_map = corr_map.reshape(batch * rows, 1, h_in, w_in)
        corr_map = F.interpolate(
            corr_map,
            size=(h_out, w_out),
            mode="bilinear",
            align_corners=True,
        )
        corr_map = corr_map.reshape(batch * rows, -1)
        corr_min = corr_map.min(dim=1, keepdim=True)[0]
        corr_max = corr_map.max(dim=1, keepdim=True)[0]
        corr_map = (corr_map - corr_min) / (corr_max - corr_min).clamp_min(1e-6)
        corr_map = corr_map > 0.5
        return corr_map.reshape(batch, rows, h_out, w_out)

    def forward(self, feature_in, logits):
        h_in, w_in = feature_in.shape[-2:]
        h_out, w_out = logits.shape[-2:]

        logits_lowres = F.interpolate(
            logits.detach(),
            size=(h_in, w_in),
            mode="bilinear",
            align_corners=True,
        )
        feature = self.proj(feature_in)
        f1 = self.conv1(feature).flatten(2)
        f2 = self.conv2(feature).flatten(2)
        logits_flat = logits_lowres.flatten(2)

        corr = torch.matmul(f1.transpose(1, 2), f2) / math.sqrt(f1.shape[1])
        corr = F.softmax(corr, dim=-1)

        corr_map_sample = self._sample_corr_rows(corr.detach())
        corr_map = self._normalize_corr_map(corr_map_sample, h_in, w_in, h_out, w_out)

        corr_out = torch.matmul(logits_flat, corr).reshape(
            logits.shape[0], logits.shape[1], h_in, w_in
        )
        corr_out = F.interpolate(
            corr_out, size=(h_out, w_out), mode="bilinear", align_corners=True
        ).contiguous()
        return {"corr_out": corr_out, "corr_map": corr_map}


class DPT_CorrMatch(DPT):
    def __init__(self, nclass=21, **kwargs):
        super().__init__(nclass=nclass, **kwargs)
        corr_in_channels = (
            self.backbone.out_channels[-1]
            if self.feature_kind == "feature_map"
            else self.backbone.embed_dim
        )
        self.corr_head = CorrPropagationHead(corr_in_channels, nclass=nclass)

    def forward(self, x, need_fp=False, feature_perturb=None, use_corr=True):
        batch_size = x.shape[0]
        features, patch_h, patch_w = self._extract_features(x)

        if feature_perturb is not None:
            features = self._apply_feature_perturbation(
                features, patch_h, patch_w, batch_size, feature_perturb
            )

        deepest = features[-1]
        if deepest.dim() == 4:
            deepest_feat = deepest.float()
        else:
            deepest_feat = (
                deepest.permute(0, 2, 1)
                .reshape(deepest.shape[0], deepest.shape[-1], patch_h, patch_w)
                .float()
                .contiguous()
            )

        if need_fp:
            features_fp = self._apply_unimatch_feature_dropout(features, patch_h, patch_w)
            features_cat = tuple(
                torch.cat((feature, feature_fp), dim=0)
                for feature, feature_fp in zip(features, features_fp)
            )
            logits = self.head(features_cat, patch_h, patch_w)
            logits = F.interpolate(
                logits, x.shape[-2:], mode="bilinear", align_corners=True
            ).contiguous()
            out, out_fp = logits.chunk(2, dim=0)
        else:
            out = self.head(features, patch_h, patch_w)
            out = F.interpolate(
                out, x.shape[-2:], mode="bilinear", align_corners=True
            ).contiguous()
            out_fp = None

        outputs = {"out": out}
        if out_fp is not None:
            outputs["out_fp"] = out_fp
        if use_corr:
            outputs.update(self.corr_head(deepest_feat, out))
        return outputs


class UPerNet_CorrMatch(UperNet):
    def __init__(self, nclass=21, **kwargs):
        super().__init__(nclass=nclass, **kwargs)
        corr_in_channels = (
            self.backbone.out_channels[-1]
            if self.feature_kind == "feature_map"
            else self.backbone.embed_dim
        )
        self.corr_head = CorrPropagationHead(corr_in_channels, nclass=nclass)

    def forward(self, x, need_fp=False, feature_perturb=None, use_corr=True):
        batch_size, _, height, width = x.shape
        feat_maps = self._extract_feature_maps(x)
        if feature_perturb is not None:
            feat_maps = tuple(
                apply_structured_feature_perturbation(feat.float(), feature_perturb)
                for feat in feat_maps
            )

        deepest_feat = feat_maps[-1].float()

        if need_fp:
            feat_maps_fp = tuple(self.fp_dropout(feat) for feat in feat_maps)
            feat_maps_cat = tuple(
                torch.cat((feat, feat_fp), dim=0)
                for feat, feat_fp in zip(feat_maps, feat_maps_fp)
            )
            pyramid_feats = self.neck(feat_maps_cat)
            logits = self.decoder(pyramid_feats)
            logits = F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            ).contiguous()
            out, out_fp = logits.chunk(2, dim=0)
        else:
            pyramid_feats = self.neck(feat_maps)
            out = self.decoder(pyramid_feats)
            out = F.interpolate(
                out, size=(height, width), mode="bilinear", align_corners=False
            ).contiguous()
            out_fp = None

        outputs = {"out": out}
        if out_fp is not None:
            outputs["out_fp"] = out_fp
        if use_corr:
            outputs.update(self.corr_head(deepest_feat, out))
        return outputs
