"""
DPT + SegMind: DPT architecture with SegMind's reconstruction head.
Combines DINOv2/v3 backbone with DPT decoder and SegMind's AuxHead_r for masked reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ConvBNReLU(nn.Sequential):
    """Conv-BN-ReLU block from SegMind."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
            nn.ReLU6(),
        )


class ConvBN(nn.Sequential):
    """Conv-BN block from SegMind."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super(ConvBN, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
        )


class AuxHead_r(nn.Module):
    """
    Reconstruction auxiliary head from SegMind.
    Reconstructs masked image regions using multi-scale features.
    """

    def __init__(self, in_channel=256, out_channel=3, indata_channel=3):
        super(AuxHead_r, self).__init__()

        self.conv4 = ConvBNReLU(in_channels=in_channel * 8 + 1, out_channels=in_channel)
        self.conv3 = ConvBNReLU(
            in_channels=in_channel * (4 + 1) + 1, out_channels=in_channel
        )
        self.conv2 = ConvBNReLU(
            in_channels=in_channel * (2 + 1) + 1, out_channels=in_channel
        )
        self.conv1 = ConvBNReLU(
            in_channels=in_channel * (1 + 1) + 1, out_channels=in_channel
        )
        self.conv0 = ConvBNReLU(
            in_channels=in_channel + 1 + indata_channel, out_channels=in_channel // 2
        )
        self.conv00 = ConvBN(
            in_channels=in_channel // 2 + 1 + indata_channel, out_channels=out_channel
        )

    def forward(self, x, res1, res2, res3, res4, h, w, mask):
        mask_small = F.interpolate(mask, size=res4.shape[-2:])
        res4 = self.conv4(torch.cat((res4, mask_small), dim=1))
        res4 = F.interpolate(res4, scale_factor=2)

        mask_small = F.interpolate(mask, size=res3.shape[-2:])
        res3 = self.conv3(torch.cat((res3, mask_small, res4), dim=1))
        res3 = F.interpolate(res3, scale_factor=2)

        mask_small = F.interpolate(mask, size=res2.shape[-2:])
        res2 = self.conv2(torch.cat((res2, mask_small, res3), dim=1))
        res2 = F.interpolate(res2, scale_factor=2)

        mask_small = F.interpolate(mask, size=res1.shape[-2:])
        res1 = self.conv1(torch.cat((res1, mask_small, res2), dim=1))
        res1 = F.interpolate(res1, scale_factor=2)

        mask_small = F.interpolate(mask, size=res1.shape[-2:])
        x_small = F.interpolate(x, size=res1.shape[-2:])
        out = self.conv0(torch.cat((res1, mask_small, x_small), dim=1))
        out = F.interpolate(out, scale_factor=2)

        mask_small = F.interpolate(mask, size=x.shape[-2:])
        out = self.conv00(torch.cat((out, mask_small, x), dim=1))

        return out


class DPTHead_SegMind(nn.Module):
    """
    DPT Head with feature output support for SegMind's reconstruction.
    """

    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead_SegMind, self).__init__()

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

        # Feature projection for auxiliary output (256 dim)
        self.feat_proj = nn.Conv2d(features, 256, kernel_size=1)

    def forward(self, out_features, patch_h, patch_w, return_features=False):
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

        seg_out = self.scratch.output_conv(path_1)

        if return_features:
            # Return intermediate features for reconstruction
            feat = self.feat_proj(path_1)
            return seg_out, feat, [layer_1, layer_2, layer_3, layer_4]

        return seg_out


class DPT_SegMind(nn.Module):
    """
    DPT model with SegMind's reconstruction auxiliary head.

    This model extends DPT architecture with:
    - DINOv2/v3 backbone
    - DPT decoder head
    - Optional AuxHead_r for masked image reconstruction

    Args:
        encoder_size: Size of the DINO encoder ('small', 'base', 'large', 'giant')
        nclass: Number of segmentation classes
        features: Number of features in DPT decoder
        out_channels: Output channels for each scale
        use_bn: Whether to use batch normalization
        backbone_version: 'dinov2' or 'dinov3'
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
        super(DPT_SegMind, self).__init__()

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

        self.head = DPTHead_SegMind(
            nclass,
            self.backbone.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
        )

        # Reconstruction head from SegMind
        # Channel multiplier based on out_channels for proper dimension matching
        self.aux_head_r = AuxHead_r(
            in_channel=out_channels[0],  # Use base channel size
            out_channel=3,
            indata_channel=3,
        )

    def lock_backbone(self):
        """Freeze backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x, mode=None, mask=None):
        """
        Forward pass.

        Args:
            x: Input image tensor [B, 3, H, W]
            mode: Forward mode
                - None: Standard segmentation output
                - 'r': Reconstruction mode, returns (seg_pred, feat, reconstructed_img)
            mask: Mask tensor for reconstruction [B, 1, H, W]

        Returns:
            - mode=None: segmentation logits [B, nclass, H, W]
            - mode='r': tuple of (seg_logits, features, reconstructed_image)
        """
        h, w = x.shape[-2:]
        patch_size = self.backbone.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size

        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )

        if mode == "r":
            # Reconstruction mode
            seg_out, feat, layer_features = self.head(
                features, patch_h, patch_w, return_features=True
            )

            # Upsample segmentation output
            seg_out = F.interpolate(
                seg_out,
                (h, w),
                mode="bilinear",
                align_corners=True,
            )

            # Reconstruct image using AuxHead_r
            layer_1, layer_2, layer_3, layer_4 = layer_features
            ah_r = nn.Tanh()(
                self.aux_head_r(x, layer_1, layer_2, layer_3, layer_4, h, w, mask)
            )

            return seg_out, feat, ah_r

        # Standard segmentation mode
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(
            out,
            (h, w),
            mode="bilinear",
            align_corners=True,
        )

        return out
