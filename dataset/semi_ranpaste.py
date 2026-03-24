from copy import deepcopy
import math
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.transform import blur, crop, hflip, normalize, obtain_cutmix_box, resize


class SemiDataset(Dataset):
    def __init__(
        self,
        name,
        root,
        mode,
        size=None,
        id_path=None,
        nsample=None,
        ignore_index=255,
        paste_cfg=None,
    ):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_index = ignore_index
        self.paste_cfg = paste_cfg or {}

        if mode in {"train_l", "train_u"}:
            with open(id_path, "r") as f:
                self.ids = f.read().splitlines()
            if mode == "train_l" and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(f"splits/{name}/val.txt", "r") as f:
                self.ids = f.read().splitlines()

    def _load_image_mask(self, sample_id):
        img = Image.open(os.path.join(self.root, sample_id.split(" ")[0])).convert("RGB")
        if self.mode == "train_u":
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask = Image.fromarray(
                np.array(Image.open(os.path.join(self.root, sample_id.split(" ")[1])))
            )
        return img, mask

    def _build_strong_view(self, img):
        strong = deepcopy(img)
        if np.random.random() < 0.8:
            strong = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(strong)
        strong = transforms.RandomGrayscale(p=0.2)(strong)
        strong = blur(strong, p=0.5)
        return strong

    def _build_paste_box(self, img_size):
        return obtain_cutmix_box(
            img_size,
            p=self.paste_cfg.get("p", 1.0),
            size_min=self.paste_cfg.get("size_min", 0.02),
            size_max=self.paste_cfg.get("size_max", 0.4),
            ratio_1=self.paste_cfg.get("ratio_1", self.paste_cfg.get("aspect_ratio_min", 0.3)),
            ratio_2=self.paste_cfg.get(
                "ratio_2", self.paste_cfg.get("aspect_ratio_max", 1 / 0.3)
            ),
        )

    def __getitem__(self, item):
        sample_id = self.ids[item]
        img, mask = self._load_image_mask(sample_id)

        if self.mode == "val":
            img, mask = normalize(img, mask)
            return img, mask, sample_id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == "train_u" else self.ignore_index
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == "train_l":
            return normalize(img, mask)

        img_w = deepcopy(img)
        img_s = self._build_strong_view(img)
        paste_box = self._build_paste_box(img_s.size[0])

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0]), dtype=np.uint8))
        img_s, ignore_mask = normalize(img_s, ignore_mask)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s, ignore_mask, paste_box

    def __len__(self):
        return len(self.ids)
