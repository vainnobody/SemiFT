import torch
import torch.nn as nn
import torch.nn.functional as F

from model.semseg.dpt import DPT
from model.semseg.corrmatch_utils import Corr


class DPT_CorrMatch(DPT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Determine deepest feature channels
        # self.head.projects is a ModuleList of 1x1 convs matching out_channels
        # The last one corresponds to the deepest feature
        deep_channels = self.head.projects[-1].in_channels

        # Projection layer as in CorrMatch DeepLabV3Plus
        self.proj = nn.Sequential(
            nn.Conv2d(
                deep_channels, 256, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        # Corr module
        # nclass is the number of output channels of the last conv in output_conv
        nclass = self.head.scratch.output_conv[2].out_channels
        self.corr = Corr(nclass=nclass)

    def forward(self, x, need_fp=False, use_corr=False):
        dict_return = {}

        patch_size = self.backbone.patch_size
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size

        intermediate_layers = self.intermediate_layer_idx[self.encoder_size]
        features = self.backbone.get_intermediate_layers(x, intermediate_layers)

        # features is a list/tuple of tensors

        # Pre-process for Corr: get deepest feature (last in list)
        feat_deepest = features[-1]

        if need_fp:
            # Feature Perturbation: Dropout and Concatenate along batch dimension
            features_expanded = []
            for f in features:
                f_drop = nn.Dropout2d(0.5)(f)
                f_cat = torch.cat(
                    (f, f_drop), dim=0
                )  # Double batch size: [Original, Perturbed]
                features_expanded.append(f_cat)

            # Forward head with expanded batch
            # Note: head expects patch_h, patch_w. Does doubling batch affect them? No.
            out_expanded = self.head(features_expanded, patch_h, patch_w)

            out_expanded = F.interpolate(
                out_expanded,
                (patch_h * patch_size, patch_w * patch_size),
                mode="bilinear",
                align_corners=True,
            )

            out, out_fp = out_expanded.chunk(2, dim=0)
            dict_return["out"] = out
            dict_return["out_fp"] = out_fp

        else:
            out = self.head(features, patch_h, patch_w)
            out = F.interpolate(
                out,
                (patch_h * patch_size, patch_w * patch_size),
                mode="bilinear",
                align_corners=True,
            )
            dict_return["out"] = out

        if use_corr:
            proj_feats = self.proj(feat_deepest)
            corr_out_dict = self.corr(proj_feats, dict_return["out"])

            dict_return["corr_map"] = corr_out_dict["corr_map"]
            # corr_out needs interpolation?
            # Corr module returns out with same spatial dim as its input 'feature_in' (which is proj_feats)
            # proj_feats is 1/14 or 1/16 of input.
            # DPT output 'out' is upsampled to input size (if self.head doesn't, we did it above).
            # Wait, Corr forward does:
            # out = F.interpolate(out.detach(), (h_in, w_in)) ...
            # ...
            # dict_return['out'] = ... (h_in, w_in) resolution

            # We need to upsample corr_out to original image size
            corr_out = corr_out_dict["out"]
            corr_out = F.interpolate(
                corr_out,
                size=(x.shape[-2], x.shape[-1]),
                mode="bilinear",
                align_corners=True,
            )

            dict_return["corr_out"] = corr_out

        return dict_return
