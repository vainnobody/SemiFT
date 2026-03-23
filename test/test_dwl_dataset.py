import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.semi_dwl import SemiDataset, TRAIN_LEN_MULTIPLIER


def _write_sample(root: Path, img_name: str, mask_name: str, fill: int):
    img = np.full((32, 32, 3), fill, dtype=np.uint8)
    mask = np.full((32, 32), fill % 3, dtype=np.uint8)
    Image.fromarray(img).save(root / img_name)
    Image.fromarray(mask).save(root / mask_name)


def test_dwl_train_length_matches_semift_sampling_style(tmp_path):
    _write_sample(tmp_path, "a_img.png", "a_mask.png", 10)
    _write_sample(tmp_path, "b_img.png", "b_mask.png", 20)
    ids = tmp_path / "train.txt"
    ids.write_text("a_img.png a_mask.png\nb_img.png b_mask.png\n")

    dataset = SemiDataset(
        "dummy",
        str(tmp_path),
        "train_u",
        size=16,
        id_path=str(ids),
        ignore_index=255,
    )

    assert len(dataset) == 2 * TRAIN_LEN_MULTIPLIER


def test_dwl_train_u_returns_weak_strong_and_valid_mask(tmp_path):
    _write_sample(tmp_path, "img.png", "mask.png", 10)
    ids = tmp_path / "train.txt"
    ids.write_text("img.png mask.png\n")

    dataset = SemiDataset(
        "dummy",
        str(tmp_path),
        "train_u",
        size=16,
        id_path=str(ids),
        ignore_index=255,
    )

    img_w, img_s, valid_mask, sample_id = dataset[0]
    assert isinstance(img_w, torch.Tensor)
    assert isinstance(img_s, torch.Tensor)
    assert valid_mask.dtype == torch.long
    assert valid_mask.shape == img_w.shape[-2:]
    assert sample_id == "img.png mask.png"


def test_dwl_val_keeps_original_length(tmp_path, monkeypatch):
    split_dir = tmp_path / "splits" / "dummy"
    split_dir.mkdir(parents=True)
    _write_sample(tmp_path, "img.png", "mask.png", 10)
    (split_dir / "val.txt").write_text("img.png mask.png\n")

    monkeypatch.chdir(tmp_path)
    dataset = SemiDataset(
        "dummy",
        str(tmp_path),
        "val",
        size=16,
        ignore_index=255,
    )

    assert len(dataset) == 1
    img, mask, sample_id = dataset[0]
    assert img.shape[-2:] == mask.shape[-2:]
    assert sample_id == "img.png mask.png"
