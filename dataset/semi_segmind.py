import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.transform import blur, crop, hflip, normalize, resize


TRAIN_LEN_MULTIPLIER = 50


def _apply_shared_geometry(img, mask, size, ignore_index, ratio_range=(0.5, 2.0)):
    """Apply the same stochastic geometric transform to an image/mask pair."""
    img, mask = resize(img, mask, ratio_range)
    img, mask = crop(img, mask, size, ignore_index)
    img, mask = hflip(img, mask, p=0.5)
    return img, mask


def _build_strong_view(img):
    strong = deepcopy(img)
    if random.random() < 0.8:
        strong = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(strong)
    strong = transforms.RandomGrayscale(p=0.2)(strong)
    strong = blur(strong, p=0.5)
    return strong


class SemiDataset(Dataset):
    """SegMind-style paired weak/strong dataset on top of SemiFT split format."""

    def __init__(
        self, name, root, mode, size=None, id_path=None, nsample=None, ignore_index=255
    ):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_index = ignore_index

        if mode in {"train_l", "train_u"}:
            with open(id_path, "r") as f:
                self.ids = f.read().splitlines()
            if mode == "train_l" and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(f"splits/{name}/val.txt", "r") as f:
                self.ids = f.read().splitlines()

    def __len__(self):
        if self.mode in {"train_l", "train_u"}:
            return len(self.ids) * TRAIN_LEN_MULTIPLIER
        return len(self.ids)

    def _get_sample_id(self, item):
        if self.mode in {"train_l", "train_u"}:
            return random.choice(self.ids)
        return self.ids[item]

    def __getitem__(self, item):
        sample_id = self._get_sample_id(item)
        img = Image.open(os.path.join(self.root, sample_id.split(" ")[0])).convert("RGB")

        if self.mode == "train_u":
            mask = Image.fromarray(
                np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
            )
        else:
            mask = Image.fromarray(
                np.array(Image.open(os.path.join(self.root, sample_id.split(" ")[1])))
            )

        if self.mode == "val":
            img, mask = normalize(img, mask)
            return img, mask, sample_id

        img, mask = _apply_shared_geometry(
            img,
            mask,
            self.size,
            self.ignore_index,
        )

        img_w = normalize(deepcopy(img))
        img_s = normalize(_build_strong_view(img))

        mask_tensor = torch.from_numpy(np.array(mask)).long()
        if self.mode == "train_l":
            return img_w, img_s, mask_tensor

        valid_mask = (mask_tensor != self.ignore_index).long()
        return img_w, img_s, valid_mask
