from copy import deepcopy
import math
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.transform import blur, crop, hflip, normalize, resize


class SegMindDataset(Dataset):
    """SegMind-style dataset.

    Labeled samples return weak/strong views with the same geometry.
    Unlabeled samples return weak/strong views plus an ignore mask.
    """

    def __init__(
        self,
        name,
        root,
        mode,
        size=None,
        id_path=None,
        nsample=None,
        ignore_index=255,
        repeat_factor=50,
    ):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_index = ignore_index
        self.repeat_factor = repeat_factor

        if mode in {"train_l", "train_u"}:
            with open(id_path, "r", encoding="utf-8") as handle:
                self.ids = handle.read().splitlines()
            if mode == "train_l" and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(f"splits/{name}/val.txt", "r", encoding="utf-8") as handle:
                self.ids = handle.read().splitlines()

        self.strong_aug = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

    def _load_pair(self, sample_id):
        img = Image.open(os.path.join(self.root, sample_id.split(" ")[0])).convert("RGB")
        if self.mode == "train_u":
            mask = Image.fromarray(
                np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
            )
        else:
            mask = Image.fromarray(
                np.array(Image.open(os.path.join(self.root, sample_id.split(" ")[1])))
            )
        return img, mask

    def _apply_shared_geom(self, img, mask):
        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == "train_u" else self.ignore_index
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)
        return img, mask

    def _strong_view(self, img):
        strong = deepcopy(img)
        strong = self.strong_aug(strong)
        strong = blur(strong, p=0.5)
        return strong

    def __getitem__(self, item):
        sample_id = self.ids[item % len(self.ids)] if self.mode == "val" else random.choice(self.ids)
        img, mask = self._load_pair(sample_id)

        if self.mode == "val":
            img, mask = normalize(img, mask)
            return img, mask, sample_id

        img, mask = self._apply_shared_geom(img, mask)
        img_w = normalize(deepcopy(img))
        img_s = normalize(self._strong_view(img))
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        if self.mode == "train_l":
            return img_w, img_s, mask_tensor

        ignore_mask = torch.zeros_like(mask_tensor)
        ignore_mask[mask_tensor == 254] = 255
        return img_w, img_s, ignore_mask

    def __len__(self):
        if self.mode == "val":
            return len(self.ids)
        return len(self.ids) * self.repeat_factor
