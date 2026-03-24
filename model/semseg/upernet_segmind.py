import torch.nn.functional as F

from model.semseg.segmind_components import ProjectionHead, ReconstructionHead
from model.semseg.upernet import UperNet


class UPerNet_SegMind(UperNet):
    def __init__(
        self,
        *args,
        fpn_channels=256,
        proj_dim=256,
        recon_channels=128,
        **kwargs,
    ):
        super().__init__(*args, fpn_channels=fpn_channels, **kwargs)
        self.proj_head = ProjectionHead(fpn_channels, proj_dim=proj_dim)
        self.recon_head = ReconstructionHead(
            fpn_channels,
            hidden_channels=recon_channels,
            out_channels=3,
        )

    def forward(self, x, return_reconstruction=False, reconstruction_mask=None):
        feat_maps = self._extract_feature_maps(x)
        pyramid_feats = self.neck(feat_maps)
        logits, decoder_feats = self.decoder(pyramid_feats, return_feats=True)
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).contiguous()

        outputs = {
            "out": logits,
            "proj_feat": self.proj_head(decoder_feats),
        }
        if return_reconstruction:
            if reconstruction_mask is None:
                raise ValueError("reconstruction_mask is required when return_reconstruction=True")
            outputs["recon"] = self.recon_head(decoder_feats, x, reconstruction_mask)
        return outputs
