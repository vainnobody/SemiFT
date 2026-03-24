import torch.nn.functional as F

from model.semseg.dpt import DPT
from model.semseg.segmind_components import ProjectionHead, ReconstructionHead


class DPT_SegMind(DPT):
    def __init__(self, *args, features=128, proj_dim=256, recon_channels=128, **kwargs):
        super().__init__(*args, features=features, **kwargs)
        self.proj_head = ProjectionHead(features, proj_dim=proj_dim)
        self.recon_head = ReconstructionHead(
            features,
            hidden_channels=recon_channels,
            out_channels=3,
        )

    def forward(
        self,
        x,
        return_proj=True,
        return_reconstruction=False,
        reconstruction_mask=None,
    ):
        features, patch_h, patch_w = self._extract_features(x)
        logits, decoder_feats = self.head(features, patch_h, patch_w, return_feats=True)
        logits = F.interpolate(
            logits,
            x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).contiguous()

        outputs = {
            "out": logits,
        }
        if return_proj:
            outputs["proj_feat"] = self.proj_head(decoder_feats)
        if return_reconstruction:
            if reconstruction_mask is None:
                raise ValueError("reconstruction_mask is required when return_reconstruction=True")
            outputs["recon"] = self.recon_head(decoder_feats, x, reconstruction_mask)
        return outputs
