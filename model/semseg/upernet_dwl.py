import torch.nn as nn
import torch.nn.functional as F

from model.semseg.upernet import UperNet


class UPerNet_DWL(UperNet):
    """DWL wrapper with an official-style pseudo classifier head."""

    def __init__(self, *args, nclass=21, fpn_channels=256, **kwargs):
        super().__init__(*args, nclass=nclass, fpn_channels=fpn_channels, **kwargs)
        self.pseudo_classifier = nn.Conv2d(fpn_channels, nclass, kernel_size=1)

    def forward(self, x, return_pseudo_pred=False):
        B, _, H, W = x.shape
        if self.feature_kind == "token":
            patch_size = self.backbone.patch_size
            patch_h, patch_w = H // patch_size, W // patch_size
            features = self.backbone.get_intermediate_layers(
                x, self.intermediate_layer_idx[self.encoder_size]
            )
        else:
            patch_h = patch_w = None
            features = self.backbone.forward_features(x)

        feat_maps = []
        for feat in features:
            if feat.dim() == 3:
                feat = feat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
            feat_maps.append(feat.float())

        pyramid_feats = self.neck(tuple(feat_maps))
        logits, decoder_feats = self.decoder(pyramid_feats, return_feats=True)
        pred = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        if not return_pseudo_pred:
            return pred

        pseudo_logits = self.pseudo_classifier(decoder_feats)
        pseudo_pred = F.interpolate(
            pseudo_logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        return pred, pseudo_pred
