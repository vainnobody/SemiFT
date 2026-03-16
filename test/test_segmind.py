import sys
import types

import torch

stub_tb = types.ModuleType("torch.utils.tensorboard")
class StubSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass
    def add_scalar(self, *args, **kwargs):
        pass
stub_tb.SummaryWriter = StubSummaryWriter
sys.modules.setdefault("torch.utils.tensorboard", stub_tb)

stub_supervised = types.ModuleType("supervised")
def _validation_cpu(*args, **kwargs):
    return 0.0, []
stub_supervised.validation_cpu = _validation_cpu
sys.modules.setdefault("supervised", stub_supervised)

from model.semseg import dpt_segmind as segmind_model
from util.segmind_utils import MemoryBank, contrastive_loss, get_batch_mask_tensor
import segmind


class DummyBackbone(torch.nn.Module):
    def __init__(self, model_name="small"):
        super().__init__()
        self.embed_dim = 32
        self.patch_size = 16
        self.weight = torch.nn.Parameter(torch.ones(1))

    def get_intermediate_layers(self, x, idx):
        b, _, h, w = x.shape
        ph, pw = h // self.patch_size, w // self.patch_size
        n = ph * pw
        return tuple(torch.randn(b, n, self.embed_dim, device=x.device) for _ in idx)


def install_dummy_backbone(monkeypatch):
    monkeypatch.setattr(segmind_model, "DINOv2", DummyBackbone)
    monkeypatch.setattr(segmind_model, "DINOv3", DummyBackbone)


def test_dpt_segmind_forward_contract(monkeypatch):
    install_dummy_backbone(monkeypatch)
    model = segmind_model.DPT_SegMind(
        encoder_size="small",
        nclass=5,
        features=32,
        out_channels=[8, 16, 32, 32],
        backbone_version="dinov2",
        proj_dim=16,
    )
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 5, 64, 64)
    logits2, proj, recon = model(x, return_aux=True)
    assert logits2.shape == (2, 5, 64, 64)
    assert proj.shape[1] == 16
    assert recon.shape == (2, 3, 64, 64)


def test_segmind_mask_tensor_shape():
    mask = get_batch_mask_tensor((2, 3, 32, 32), mask_gap=4, mask_rate=0.25)
    assert mask.shape == (2, 1, 32, 32)
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})


def test_contrastive_loss_accepts_empty_bank():
    feat = torch.randn(2, 8, 8, 8)
    lab = torch.randint(0, 3, (2, 8, 8))
    prob = torch.softmax(torch.randn(2, 3, 8, 8), dim=1)
    bank = MemoryBank(class_num=3, bank_size=16, feat_dim=8)
    opt = {
        "class_num": 3,
        "query_threshold": 0.97,
        "num_query": 4,
        "num_negative": 4,
        "temperature": 0.5,
    }
    loss = contrastive_loss(feat, lab, prob, opt, bank)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0


def test_segmind_source_has_reconstruction_and_contrastive_terms():
    source = open("segmind.py", "r", encoding="utf-8").read()
    assert 'contrastive_loss' in source
    assert 'get_batch_mask_tensor' in source
    assert 'mode="r"' in source
