import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3
from model.backbone.resnet import ResNet101Backbone
from model.semseg.feature_perturb import apply_structured_feature_perturbation
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

    def forward(self, out_features, patch_h, patch_w):
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

        self.head = DPTHead(
            nclass,
            self.backbone.out_channels if self.feature_kind == "feature_map" else self.backbone.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            feature_kind=self.feature_kind,
        )

        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

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

    def forward(self, x, comp_drop=False, feature_perturb=None):
        batch_size = x.shape[0]
        features, patch_h, patch_w = self._extract_features(x)

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

        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, x.shape[-2:], mode="bilinear", align_corners=True)

        return out
