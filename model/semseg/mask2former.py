# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from Mask2Former and DPT for semantic segmentation with DINOv2/v3 backbone.

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3
from model.util.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    """Create a feature fusion block for DPT-style progressive upsampling."""
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class PositionEmbeddingSine(nn.Module):
    """
    2D sinusoidal position embedding, adapted from Mask2Former.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class SelfAttentionLayer(nn.Module):
    """Self-attention layer from Mask2Former Transformer Decoder."""

    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(
                q,
                k,
                value=tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout(tgt2)
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(
                q,
                k,
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
        return tgt


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer from Mask2Former Transformer Decoder."""

    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout(tgt2)
        else:
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
        return tgt


class FFNLayer(nn.Module):
    """Feed-forward network layer from Mask2Former Transformer Decoder."""

    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout(tgt2)
        else:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Simple multi-layer perceptron (FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DPTStylePixelDecoder(nn.Module):
    """
    DPT-style pixel decoder that fuses multi-scale features from DINOv2/v3.

    This replaces the MSDeformAttn pixel decoder from original Mask2Former
    with a simpler DPT-style progressive feature fusion.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        use_bn: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim

        # Project backbone features to different channel dimensions
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

        # Resize layers to create multi-scale features
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

        # DPT-style scratch layers for feature refinement
        self.scratch = _make_scratch(out_channels, hidden_dim, groups=1, expand=False)
        self.scratch.stem_transpose = None

        # Refinement networks for progressive fusion
        self.scratch.refinenet1 = _make_fusion_block(hidden_dim, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(hidden_dim, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(hidden_dim, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(hidden_dim, use_bn)

        # Final mask features projection
        self.mask_features_proj = nn.Conv2d(hidden_dim, mask_dim, kernel_size=1)

        # Multi-scale output projections for transformer decoder
        self.multi_scale_projs = nn.ModuleList(
            [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            ]
        )

    def forward(
        self, features: List[torch.Tensor], patch_h: int, patch_w: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            features: List of 4 feature tensors from backbone [B, N, C]
            patch_h, patch_w: Height and width in patches

        Returns:
            mask_features: [B, mask_dim, H, W] for mask prediction
            multi_scale_features: List of 3 features for transformer decoder
        """
        out = []
        for i, x in enumerate(features):
            # Reshape from [B, N, C] to [B, C, H, W]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        # Apply scratch layers
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # Progressive fusion (top-down path)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Generate mask features from highest resolution
        mask_features = self.mask_features_proj(path_1)

        # Generate multi-scale features for transformer decoder
        # Use 3 scales: low, medium, high resolution
        multi_scale_features = [
            self.multi_scale_projs[0](path_3),  # Low-res
            self.multi_scale_projs[1](path_2),  # Medium-res
            self.multi_scale_projs[2](path_1),  # High-res
        ]

        return mask_features, multi_scale_features


class Mask2FormerTransformerDecoder(nn.Module):
    """
    Mask2Former Transformer Decoder for semantic segmentation.

    Uses learnable queries and masked cross-attention to predict
    class labels and mask embeddings.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        pre_norm: bool = False,
        mask_dim: int = 256,
        num_feature_levels: int = 3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_feature_levels = num_feature_levels

        # Positional encoding for features
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # Transformer decoder layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Learnable query features and positional embeddings
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Level embedding for multi-scale features
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # Input projections for each feature level
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            self.input_proj.append(
                nn.Sequential()
            )  # Already projected in pixel decoder

        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(
        self, multi_scale_features: List[torch.Tensor], mask_features: torch.Tensor
    ) -> dict:
        """
        Args:
            multi_scale_features: List of 3 multi-scale features [B, C, H, W]
            mask_features: [B, mask_dim, H, W]

        Returns:
            dict with 'pred_logits' and 'pred_masks'
        """
        assert len(multi_scale_features) == self.num_feature_levels

        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            pos.append(self.pe_layer(multi_scale_features[i], None).flatten(2))
            src.append(
                self.input_proj[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )
            # Flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # Initialize queries: QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # Initial prediction
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # Prevent attending to all masked positions
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # Cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed,
            )

            # Self-attention
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            # Prediction at each layer
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        return {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(predictions_class, predictions_mask),
        }

    def forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate class predictions and mask predictions."""
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [Q, B, C] -> [B, Q, C]

        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # Generate attention mask for next layer
        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        """Set auxiliary losses for deep supervision."""
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]


class Mask2FormerHead(nn.Module):
    """
    Complete Mask2Former head combining pixel decoder and transformer decoder.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        pre_norm: bool = False,
        out_channels: List[int] = [256, 512, 1024, 1024],
        use_bn: bool = False,
    ):
        super().__init__()

        self.pixel_decoder = DPTStylePixelDecoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            mask_dim=mask_dim,
            out_channels=out_channels,
            use_bn=use_bn,
        )

        self.transformer_decoder = Mask2FormerTransformerDecoder(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
        )

        self.num_classes = num_classes

    def forward(self, features: List[torch.Tensor], patch_h: int, patch_w: int) -> dict:
        """
        Args:
            features: List of 4 backbone features [B, N, C]
            patch_h, patch_w: Patch grid size

        Returns:
            dict with predictions
        """
        mask_features, multi_scale_features = self.pixel_decoder(
            features, patch_h, patch_w
        )
        outputs = self.transformer_decoder(multi_scale_features, mask_features)
        return outputs


class Mask2Former(nn.Module):
    """
    Mask2Former with DPT-style pixel decoder and DINOv2/v3 backbone.

    This implementation combines:
    - DINOv2 or DINOv3 as the backbone (pretrained ViT)
    - DPT-style multi-scale feature fusion
    - Mask2Former transformer decoder for semantic segmentation
    """

    def __init__(
        self,
        encoder_size: str = "base",
        nclass: int = 21,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        out_channels: List[int] = [256, 512, 1024, 1024],
        use_bn: bool = False,
        backbone_version: str = "dinov2",
    ):
        super().__init__()

        # Layer indices for intermediate feature extraction
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
        self.nclass = nclass

        # Initialize backbone
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

        # Initialize Mask2Former head
        self.head = Mask2FormerHead(
            num_classes=nclass,
            in_channels=self.backbone.embed_dim,
            hidden_dim=hidden_dim,
            mask_dim=mask_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            out_channels=out_channels,
            use_bn=use_bn,
        )

    def lock_backbone(self):
        """Freeze backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for semantic segmentation.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Semantic segmentation logits [B, nclass, H, W]
        """
        patch_size = self.backbone.patch_size
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size

        # Extract multi-scale features from backbone
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )

        # Get mask predictions
        outputs = self.head(features, patch_h, patch_w)

        # Semantic segmentation inference
        mask_cls = outputs["pred_logits"]  # [B, Q, nclass+1]
        mask_pred = outputs["pred_masks"]  # [B, Q, H, W]

        # Semantic inference: combine class probs and mask probs
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # Remove no-object class
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)

        # Upsample to input resolution
        semseg = F.interpolate(
            semseg,
            size=(patch_h * patch_size, patch_w * patch_size),
            mode="bilinear",
            align_corners=True,
        )

        return semseg

    def forward_with_aux(self, x: torch.Tensor) -> dict:
        """
        Forward pass returning full outputs including auxiliary predictions.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            dict with 'pred_logits', 'pred_masks', 'aux_outputs'
        """
        patch_size = self.backbone.patch_size
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size

        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )

        outputs = self.head(features, patch_h, patch_w)
        return outputs
