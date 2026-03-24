from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.semseg.dpt import DPT
from model.semseg.upernet import UperNet


class SegMindModel(nn.Module):
    def __init__(self, base_model: nn.Module, nclass: int, project_dim: int = 256):
        super().__init__()
        self.base_model = base_model
        self.nclass = nclass
        self.project_dim = project_dim

        if isinstance(base_model, DPT):
            feat_dim = base_model.head.scratch.output_conv[0].out_channels
        elif isinstance(base_model, UperNet):
            feat_dim = base_model.decoder.classifier.in_channels
        else:
            raise TypeError(f"Unsupported base model type: {type(base_model)!r}")

        self.projector = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, project_dim, kernel_size=1),
        )
        self.recon_head = nn.Sequential(
            nn.Conv2d(feat_dim + 4, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(feat_dim // 2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(feat_dim // 2, 1), 3, kernel_size=1),
            nn.Tanh(),
        )
        self.mask_classifier = nn.Conv2d(project_dim, nclass, kernel_size=1)

    @property
    def backbone(self):
        return self.base_model.backbone

    @property
    def head(self):
        return self.base_model.head

    def lock_backbone(self):
        return self.base_model.lock_backbone()

    def _forward_features(self, x: torch.Tensor):
        if isinstance(self.base_model, DPT):
            features, patch_h, patch_w = self.base_model._extract_features(x)
            seg_logits, decoder_feat = self.base_model.head(features, patch_h, patch_w, return_feats=True)
            seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
            decoder_feat = F.interpolate(
                decoder_feat,
                size=(x.shape[-2] // 4, x.shape[-1] // 4),
                mode="bilinear",
                align_corners=True,
            )
            return seg_logits, decoder_feat

        feat_maps = self.base_model._extract_feature_maps(x)
        pyramid_feats = self.base_model.neck(feat_maps)
        seg_logits, decoder_feat = self.base_model.decoder(pyramid_feats, return_feats=True)
        seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return seg_logits, decoder_feat

    def forward(self, x: torch.Tensor, mim_mask: torch.Tensor | None = None, return_aux: bool = False):
        seg_logits, decoder_feat = self._forward_features(x)
        if not return_aux and mim_mask is None:
            return seg_logits

        proj_feat = self.projector(decoder_feat)
        outputs = {
            "seg_logits": seg_logits,
            "proj_feat": proj_feat,
            "decoder_feat": decoder_feat,
        }

        if mim_mask is not None:
            low_mask = F.interpolate(mim_mask.float(), size=decoder_feat.shape[-2:], mode="nearest")
            low_img = F.interpolate(x, size=decoder_feat.shape[-2:], mode="bilinear", align_corners=False)
            recon_in = torch.cat((decoder_feat, low_img, low_mask), dim=1)
            recon_img = self.recon_head(recon_in)
            outputs["recon_img"] = recon_img
            outputs["mask_logits"] = self.mask_classifier(proj_feat)

        if return_aux or mim_mask is not None:
            return outputs
        return seg_logits
