import json
from pathlib import Path

import numpy as np
import torch

import CKA as cka_mod


class DummyDataset:
    def __init__(self, ids, name="potsdam"):
        self.ids = list(ids)
        self.name = name

    def __len__(self):
        return len(self.ids)


def test_linear_cka_self_similarity_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 16))
    score = cka_mod.linear_cka(x, x)
    assert np.isclose(score, 1.0, atol=1e-6)


def test_linear_cka_is_symmetric():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 12))
    y = rng.normal(size=(40, 8))
    score_xy = cka_mod.linear_cka(x, y)
    score_yx = cka_mod.linear_cka(y, x)
    assert np.isclose(score_xy, score_yx, atol=1e-10)


def test_extract_state_dict_prefers_model_ema_in_auto_mode():
    checkpoint = {
        "model": {"module.backbone.weight": torch.ones(1)},
        "model_ema": {"module.backbone.weight": torch.zeros(1)},
    }
    state_dict, source = cka_mod.extract_state_dict(checkpoint, source="auto")
    assert source == "model_ema"
    assert torch.equal(state_dict["module.backbone.weight"], torch.zeros(1))


def test_strip_known_prefixes_removes_module_prefix():
    state_dict = {
        "module.backbone.weight": torch.ones(2),
        "module.head.bias": torch.zeros(2),
    }
    stripped = cka_mod.strip_known_prefixes(state_dict)
    assert set(stripped.keys()) == {"backbone.weight", "head.bias"}


def test_create_sample_manifest_is_deterministic():
    dataset = DummyDataset([f"img_{i}.png mask_{i}.png" for i in range(10)])
    manifest = cka_mod.create_sample_manifest(dataset, split="val", max_samples=3, sample_stride=2)
    assert [entry["dataset_index"] for entry in manifest["entries"]] == [0, 2, 4]
    assert manifest["entries"][1]["sample_key"].startswith("00002_")


def test_validate_manifest_against_dataset_rejects_mismatch():
    dataset = DummyDataset(["a b", "c d"])
    manifest = {
        "dataset": "potsdam",
        "split": "val",
        "entries": [{"dataset_index": 0, "sample_id": "wrong id", "sample_key": "00000_wrong"}],
    }
    try:
        cka_mod.validate_manifest_against_dataset(manifest, dataset, "val")
    except ValueError as exc:
        assert "Manifest sample mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched manifest")


def test_select_zoom_boxes_finds_improvement_region():
    gt = np.zeros((32, 32), dtype=np.int32)
    baseline = np.ones((32, 32), dtype=np.int32)
    current = np.ones((32, 32), dtype=np.int32)
    current[10:16, 12:20] = 0
    boxes = cka_mod.select_zoom_boxes(baseline, current, gt, ignore_index=255, max_boxes=2, pad=0, min_component_area=4)
    assert boxes
    x1, y1, x2, y2 = boxes[0]
    assert x1 <= 12 < x2 and y1 <= 10 < y2
    assert x1 < 20 <= x2 and y1 < 16 <= y2


def test_manifests_match_requires_same_entries():
    manifest_a = {"dataset": "potsdam", "split": "val", "entries": [{"dataset_index": 0, "sample_id": "a", "sample_key": "00000_a"}]}
    manifest_b = {"dataset": "potsdam", "split": "val", "entries": [{"dataset_index": 0, "sample_id": "a", "sample_key": "00000_a"}]}
    manifest_c = {"dataset": "potsdam", "split": "val", "entries": [{"dataset_index": 1, "sample_id": "b", "sample_key": "00001_b"}]}
    assert cka_mod.manifests_match(manifest_a, manifest_b)
    assert not cka_mod.manifests_match(manifest_a, manifest_c)
