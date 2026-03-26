import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_channels, proj_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, proj_dim, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return F.normalize(self.layers(x), dim=1)


class ReconstructionHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels + out_channels + 1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, decoder_feat, masked_input, reconstruction_mask):
        feat = F.interpolate(
            decoder_feat,
            size=masked_input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.layers(
            torch.cat((feat, masked_input, reconstruction_mask.float()), dim=1)
        )
