import sys
import types
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

stub_tb = types.ModuleType("torch.utils.tensorboard")
stub_einops = types.ModuleType("einops")


class StubSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass


stub_tb.SummaryWriter = StubSummaryWriter
stub_einops.rearrange = lambda x, *args, **kwargs: x
sys.modules.setdefault("torch.utils.tensorboard", stub_tb)
sys.modules.setdefault("einops", stub_einops)

from model.semseg.dpt import DPT
from model.semseg.dpt_segmind import DPT_SegMind
from model.semseg.upernet import UperNet
from model.semseg.upernet_segmind import UPerNet_SegMind
from util.segmind_utils import (
    classmix_batch,
    compute_contrastive_loss,
    get_batch_mask_tensor,
    init_queue_state,
    load_queue_state,
    serialize_queue_state,
)


class _FakeDPTHead(nn.Module):
    def forward(self, features, patch_h, patch_w, return_feats=False):
        batch = features[0].shape[0]
        logits = torch.randn(batch, 5, patch_h, patch_w)
        feats = torch.randn(batch, 8, patch_h, patch_w)
        if return_feats:
            return logits, feats
        return logits


class _FakeDecoder(nn.Module):
    def forward(self, feats, return_feats=False):
        batch = feats[0].shape[0]
        logits = torch.randn(batch, 5, 4, 4)
        decoder_feats = torch.randn(batch, 8, 4, 4)
        if return_feats:
            return logits, decoder_feats
        return logits


def _fake_dpt_init(self, *args, **kwargs):
    nn.Module.__init__(self)
    self.head = _FakeDPTHead()


def _fake_upernet_init(self, *args, **kwargs):
    nn.Module.__init__(self)
    self.neck = nn.Identity()
    self.decoder = _FakeDecoder()


def test_dpt_segmind_wrapper_returns_expected_keys():
    with mock.patch.object(DPT, "__init__", _fake_dpt_init):
        model = DPT_SegMind(features=8, proj_dim=4, recon_channels=6)
    model._extract_features = lambda x: ((torch.randn(x.shape[0], 16, 8),), 4, 4)

    outputs = model(torch.randn(2, 3, 16, 16))
    recon_outputs = model(
        torch.randn(2, 3, 16, 16),
        return_reconstruction=True,
        reconstruction_mask=torch.ones(2, 1, 16, 16),
    )

    assert set(outputs.keys()) == {"out", "proj_feat"}
    assert outputs["out"].shape == (2, 5, 16, 16)
    assert outputs["proj_feat"].shape == (2, 4, 4, 4)
    assert "recon" in recon_outputs
    assert recon_outputs["recon"].shape == (2, 3, 16, 16)
    outputs_no_proj = model(torch.randn(2, 3, 16, 16), return_proj=False)
    assert set(outputs_no_proj.keys()) == {"out"}


def test_upernet_segmind_wrapper_returns_expected_keys():
    with mock.patch.object(UperNet, "__init__", _fake_upernet_init):
        model = UPerNet_SegMind(fpn_channels=8, proj_dim=4, recon_channels=6)
    model._extract_feature_maps = lambda x: (torch.randn(x.shape[0], 8, 4, 4),) * 4

    outputs = model(torch.randn(2, 3, 16, 16))
    recon_outputs = model(
        torch.randn(2, 3, 16, 16),
        return_reconstruction=True,
        reconstruction_mask=torch.ones(2, 1, 16, 16),
    )

    assert set(outputs.keys()) == {"out", "proj_feat"}
    assert outputs["out"].shape == (2, 5, 16, 16)
    assert outputs["proj_feat"].shape == (2, 4, 4, 4)
    assert "recon" in recon_outputs
    assert recon_outputs["recon"].shape == (2, 3, 16, 16)
    outputs_no_proj = model(torch.randn(2, 3, 16, 16), return_proj=False)
    assert set(outputs_no_proj.keys()) == {"out"}


def test_segmind_queue_state_roundtrip_and_contrastive_loss():
    queue_state = init_queue_state(num_classes=3, feat_dim=4, bank_size=16)
    proj_feat = torch.randn(2, 4, 4, 4)
    labels = torch.randint(0, 3, (2, 8, 8))
    probs = torch.softmax(torch.randn(2, 3, 8, 8), dim=1)

    loss = compute_contrastive_loss(
        proj_feat,
        labels,
        probs,
        queue_state,
        query_threshold=0.95,
        temperature=0.5,
        num_query=4,
        num_negative=4,
        ignore_index=255,
    )
    payload = serialize_queue_state(queue_state)
    restored = load_queue_state(payload)

    assert torch.isfinite(loss)
    assert restored.bank_size == 16
    assert len(restored.banks) == 3


def test_segmind_block_masks_match_requested_shape():
    masks = get_batch_mask_tensor((3, 3, 16, 16), mask_gap=4, mask_rate=0.25)
    assert masks.shape == (3, 1, 16, 16)
    unique_values = set(torch.unique(masks).tolist())
    assert unique_values.issubset({0.0, 1.0})


def test_classmix_batch_preserves_batch_tensor_ranks():
    img_u_w = torch.randn(2, 3, 8, 8)
    img_u_s = torch.randn(2, 3, 8, 8)
    pseudo_label = torch.randint(0, 5, (2, 8, 8))
    pseudo_conf = torch.rand(2, 8, 8)
    entropy = torch.rand(2, 8, 8)
    valid = torch.ones(2, 8, 8)

    outputs = classmix_batch(
        img_u_w,
        img_u_s,
        pseudo_label.float(),
        pseudo_conf,
        entropy,
        valid,
        labels=pseudo_label,
    )

    assert outputs[0].shape == (2, 3, 8, 8)
    assert outputs[1].shape == (2, 3, 8, 8)
    assert outputs[2].shape == (2, 8, 8)
    assert outputs[6].shape == (2, 8, 8)


def test_contrastive_loss_keeps_gradient_on_query_features():
    queue_state = init_queue_state(num_classes=3, feat_dim=4, bank_size=16)
    proj_feat = torch.randn(2, 4, 4, 4, requires_grad=True)
    labels = torch.randint(0, 3, (2, 8, 8))
    probs = torch.softmax(torch.randn(2, 3, 8, 8), dim=1)

    loss = compute_contrastive_loss(
        proj_feat,
        labels,
        probs,
        queue_state,
        query_threshold=0.95,
        temperature=0.5,
        num_query=4,
        num_negative=4,
        ignore_index=255,
    )
    loss.backward()

    assert proj_feat.grad is not None
    assert torch.isfinite(proj_feat.grad).all()
