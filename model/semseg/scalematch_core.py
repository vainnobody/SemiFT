import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2_layers.drop_path import DropPath


def scale_as(x, target, align_corners=True):
    if x.shape[-2:] == target.shape[-2:]:
        return x
    return F.interpolate(
        x,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=align_corners,
    )


def resize_x(x, scale_factor, patch_size=14, align_corners=True):
    if scale_factor == 1.0:
        return x

    h, w = x.shape[-2:]
    target_h = max(round((h * scale_factor) / patch_size) * patch_size, patch_size)
    target_w = max(round((w * scale_factor) / patch_size) * patch_size, patch_size)
    return F.interpolate(
        x,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=align_corners,
    )


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
        for block in self.blocks:
            x = block(x)
        return self.norm(x), h, w
