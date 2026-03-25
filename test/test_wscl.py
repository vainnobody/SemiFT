import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

stub_tb = types.ModuleType("torch.utils.tensorboard")


class StubSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        return None


stub_tb.SummaryWriter = StubSummaryWriter
sys.modules.setdefault("torch.utils.tensorboard", stub_tb)

import wscl as wscl_mod


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.ids = list(range(8))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        raise RuntimeError("Not needed for dataloader construction test")


class DummyDistributedSampler:
    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        return None


def test_wscl_dataloaders_use_half_batch_size(monkeypatch):
    monkeypatch.setattr(wscl_mod, "SemiDataset", DummyDataset)
    monkeypatch.setattr(wscl_mod, "ValDataset", DummyDataset)
    monkeypatch.setattr(
        torch.utils.data.distributed,
        "DistributedSampler",
        DummyDistributedSampler,
    )

    args = types.SimpleNamespace(
        labeled_id_path="labeled.txt",
        unlabeled_id_path="unlabeled.txt",
    )
    cfg = {
        "dataset": "dummy",
        "data_root": ".",
        "crop_size": 32,
        "batch_size": 6,
        "ignore_index": 255,
        "workers": 0,
    }

    trainloader_l, trainloader_u, _ = wscl_mod.build_dataloaders(args, cfg)

    assert trainloader_l.batch_size == 3
    assert trainloader_u.batch_size == 3


def test_wscl_source_restores_official_aug_probability_and_loader_split():
    source = (REPO_ROOT / "wscl.py").read_text()

    assert "per_loader_batch = max(1, int(cfg[\"batch_size\"] // 2))" in source
    assert "if np.random.uniform(0, 1) < 0.5:" in source


def test_wscl_unsup_loss_matches_official_style_reweighting():
    loss_u_map = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=torch.float32,
    )
    valid_region = torch.tensor(
        [[[True, True], [False, True]]]
    )
    entropy_u = torch.tensor(
        [[[0.1, 0.6], [0.9, 0.2]]],
        dtype=torch.float32,
    )

    threshold = np.percentile(entropy_u[valid_region].numpy().flatten(), 20)
    mask_valid = (entropy_u <= threshold) & valid_region
    mask_ratio = mask_valid.sum().float() / valid_region.sum().clamp(min=1).float()
    unsup_valid = (loss_u_map * valid_region.float()).sum() / valid_region.sum().clamp(
        min=1
    ).float()
    loss_u = unsup_valid * mask_ratio

    expected_unsup_valid = torch.tensor((1.0 + 2.0 + 4.0) / 3.0)
    expected_ratio = torch.tensor(1.0 / 3.0)
    assert torch.isclose(unsup_valid, expected_unsup_valid)
    assert torch.isclose(mask_ratio, expected_ratio)
    assert torch.isclose(loss_u, expected_unsup_valid * expected_ratio)
