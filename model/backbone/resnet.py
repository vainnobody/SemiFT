import torch.nn as nn


class ResNet101Backbone(nn.Module):
    feature_kind = "feature_map"
    out_channels = [256, 512, 1024, 2048]
    output_stride = 32
    patch_size = 32
    embed_dim = 2048

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import resnet101
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise ImportError(
                "torchvision is required to use the ResNet-101 backbone."
            ) from exc

        model = resnet101(weights=None)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def load_state_dict(self, state_dict, strict=True):
        """Load torchvision-style ResNet-101 checkpoints.

        Official torchvision checkpoints include classifier weights (`fc.*`) that are
        not part of the backbone-only wrapper used by SemiFT. Accept and drop them so
        callers can pass the raw torchvision state dict directly.
        """
        if isinstance(state_dict, dict):
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k not in {"fc.weight", "fc.bias"}
            }
        return super().load_state_dict(state_dict, strict=strict)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return (c2, c3, c4, c5)

    def forward(self, x):
        return self.forward_features(x)
