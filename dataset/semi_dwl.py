from copy import deepcopy
import math
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.transform import blur, crop, hflip, normalize, resize


TRAIN_LEN_MULTIPLIER = 50


def vflip(image, label, p=0.5):
    if random.random() < p:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
    return image, label


def rotate(image, label):
    angle = random.choice([90, 180, 270, 360])
    image = image.rotate(angle, expand=True)
    label = label.rotate(angle, expand=True)
    return image, label


class SemiDataset(Dataset):
    """DWL-specific dataset with SemiFT-style epoch length semantics."""

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
            with open("splits/%s/val.txt" % name, "r") as f:
                self.ids = f.read().splitlines()

    def __len__(self):
        if self.mode in {"train_l", "train_u"}:
            return len(self.ids) * TRAIN_LEN_MULTIPLIER
        return len(self.ids)

    def __getitem__(self, item):
        sample_id = (
            random.choice(self.ids) if self.mode in {"train_l", "train_u"} else self.ids[item]
        )
        img = Image.open(os.path.join(self.root, sample_id.split(" ")[0])).convert("RGB")
        mask = Image.fromarray(
            np.array(Image.open(os.path.join(self.root, sample_id.split(" ")[1])))
        )

        if self.mode == "val":
            img, mask = normalize(img, mask)
            return img, mask, sample_id

        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size, self.ignore_index)
        img, mask = hflip(img, mask, p=0.5)
        img, mask = vflip(img, mask, p=0.5)
        img, mask = rotate(img, mask)

        if self.mode == "train_l":
            return normalize(img, mask)

        img_w = normalize(deepcopy(img))
        img_s = deepcopy(img)
        if random.random() < 0.8:
            img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
        img_s = transforms.RandomGrayscale(p=0.2)(img_s)
        img_s = blur(img_s, p=0.5)
        img_s, valid_mask = normalize(img_s, mask)
        valid_mask = (valid_mask != self.ignore_index).long()
        return img_w, img_s, valid_mask, sample_id
