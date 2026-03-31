import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleMatchLogitFusion(nn.Module):
    """Lightweight logit fusion head for ScaleMatch-style joint prediction."""

    def __init__(self, nclass: int):
        super().__init__()
        hidden = max(nclass, 16)
        self.net = nn.Sequential(
            nn.Conv2d(nclass * 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
        )

    def forward(self, pred_ori: torch.Tensor, pred_scale: torch.Tensor) -> torch.Tensor:
        weights = self.net(torch.cat((pred_ori, pred_scale), dim=1)).softmax(dim=1)
        return (
            weights[:, :1] * pred_ori
            + weights[:, 1:] * pred_scale
        )


class ScaleMatchModel(nn.Module):
    """Wrap an existing semseg model with official-ScaleMatch-style two-scale outputs."""

    def __init__(self, model: nn.Module, nclass: int):
        super().__init__()
        self.model = model
        self.fusion = ScaleMatchLogitFusion(nclass)

    @property
    def backbone(self):
        return self.model.backbone

    @property
    def head(self):
        return self.model.head

    def lock_backbone(self):
        if hasattr(self.model, "lock_backbone"):
            self.model.lock_backbone()

    def _resize_input(self, x: torch.Tensor, scale_factor: float) -> torch.Tensor:
        if scale_factor == 1.0:
            return x

        h, w = x.shape[-2:]
        target_h = max(1, int(round(h * scale_factor)))
        target_w = max(1, int(round(w * scale_factor)))

        patch_size = getattr(self.backbone, "patch_size", None)
        if patch_size is not None:
            target_h = max(patch_size, int(round(target_h / patch_size)) * patch_size)
            target_w = max(patch_size, int(round(target_w / patch_size)) * patch_size)

        return F.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=True
        )

    def _forward_base(self, x: torch.Tensor, need_fp: bool = False, **kwargs):
        return self.model(x, need_fp=need_fp, **kwargs)

    def forward(
        self,
        inputs: torch.Tensor,
        scale_factor=None,
        feature_scale=1.0,
        scales=None,
        eval_mode="atten_fusion",
        plain_inputs: torch.Tensor = None,
        **kwargs,
    ):
        del feature_scale, eval_mode  # kept for official-API compatibility

        if scales:
            preds = []
            for scale in scales:
                scaled_inputs = self._resize_input(inputs, float(scale))
                pred = self._forward_base(scaled_inputs, **kwargs)
                pred = F.interpolate(
                    pred,
                    size=inputs.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                preds.append(pred)
            return torch.stack(preds, dim=0).mean(dim=0)

        if scale_factor is None:
            return self._forward_base(inputs, **kwargs)

        scale_factor = float(scale_factor)
        pred_plain = self._forward_base(plain_inputs, **kwargs) if plain_inputs is not None else None
        if math.isclose(scale_factor, 1.0):
            pred_ori, pred_fp = self._forward_base(inputs, need_fp=True, **kwargs)
            return {
                "pred_joint": pred_ori,
                "pred_ori": pred_ori,
                "pred_fp": pred_fp,
                "pred_size": pred_ori,
                "pred_plain": pred_plain,
                "out": pred_ori,
            }

        if scale_factor > 1.0:
            pred_ori, pred_fp = self._forward_base(inputs, need_fp=True, **kwargs)
            pred_scale = self._forward_base(
                self._resize_input(inputs, scale_factor), **kwargs
            )
            pred_scale = F.interpolate(
                pred_scale,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            pred_joint = self.fusion(pred_ori, pred_scale)
            return {
                "pred_joint": pred_joint,
                "pred_ori": pred_ori,
                "pred_fp": pred_fp,
                "pred_size": pred_scale,
                "pred_plain": pred_plain,
                "out": pred_joint,
            }

        pred_scale = self._forward_base(self._resize_input(inputs, scale_factor), **kwargs)
        pred_scale = F.interpolate(
            pred_scale,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        pred_ori, pred_fp = self._forward_base(inputs, need_fp=True, **kwargs)
        pred_joint = self.fusion(pred_ori, pred_scale)
        return {
            "pred_joint": pred_joint,
            "pred_ori": pred_ori,
            "pred_fp": pred_fp,
            "pred_size": pred_scale,
            "pred_plain": pred_plain,
            "out": pred_joint,
        }
