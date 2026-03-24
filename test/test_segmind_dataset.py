import random
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.semi_segmind import SemiDataset


def _write_sample(root, stem, size=(16, 16), color=32):
    img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    mask = np.full((size[1], size[0]), 2, dtype=np.uint8)
    img_path = root / f"{stem}.png"
    mask_path = root / f"{stem}_mask.png"
    Image.fromarray(img).save(img_path)
    Image.fromarray(mask).save(mask_path)
    return img_path.name, mask_path.name


def test_segmind_labeled_dataset_returns_paired_views_with_shared_geometry():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        img_name, mask_name = _write_sample(root, "sample")
        ids = root / "labeled.txt"
        ids.write_text(f"{img_name} {mask_name}\n", encoding="utf-8")

        dataset = SemiDataset(
            "pascal",
            str(root),
            "train_l",
            size=8,
            id_path=str(ids),
            ignore_index=255,
        )

        with mock.patch(
            "dataset.semi_segmind._build_strong_view",
            side_effect=lambda image: image.copy(),
        ):
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            img_w, img_s, mask = dataset[0]

        assert img_w.shape == img_s.shape == (3, 8, 8)
        assert torch.equal(img_w, img_s)
        assert mask.shape == (8, 8)
        assert mask.dtype == torch.long


def test_segmind_unlabeled_dataset_marks_padded_region_invalid():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        img_name, mask_name = _write_sample(root, "sample", size=(4, 4))
        ids = root / "unlabeled.txt"
        ids.write_text(f"{img_name} {mask_name}\n", encoding="utf-8")

        dataset = SemiDataset(
            "pascal",
            str(root),
            "train_u",
            size=8,
            id_path=str(ids),
            ignore_index=255,
        )

        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        img_w, img_s, valid_mask = dataset[0]

        assert img_w.shape == img_s.shape == (3, 8, 8)
        assert valid_mask.shape == (8, 8)
        assert valid_mask.dtype == torch.long
        assert valid_mask.sum().item() < valid_mask.numel()
