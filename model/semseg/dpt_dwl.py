from copy import deepcopy

import torch.nn.functional as F

from model.semseg.dpt import DPT


class DPT_DWL(DPT):
    """DWL wrapper with a dedicated pseudo head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pseudo_head = deepcopy(self.head)

    def forward(self, x, return_pseudo_pred=False):
        features, patch_h, patch_w = self._extract_features(x)
        pred = self.head(features, patch_h, patch_w)
        pred = F.interpolate(
            pred,
            x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        if not return_pseudo_pred:
            return pred

        pseudo_pred = self.pseudo_head(features, patch_h, patch_w)
        pseudo_pred = F.interpolate(
            pseudo_pred,
            x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        return pred, pseudo_pred
