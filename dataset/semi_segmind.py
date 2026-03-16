from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.transform import blur, crop, hflip, normalize, resize


class SemiDataset(Dataset):
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

    def __getitem__(self, item):
        sample_id = random.choice(self.ids) if self.mode in {"train_l", "train_u"} else self.ids[item]
        img = Image.open(os.path.join(self.root, sample_id.split(" ")[0])).convert("RGB")
        if self.mode == "train_u":
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, sample_id.split(" ")[1]))))

        if self.mode == "val":
            img, mask = normalize(img, mask)
            return img, mask, sample_id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == "train_u" else self.ignore_index
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        img_w = deepcopy(img)
        img_s = deepcopy(img)
        if random.random() < 0.8:
            img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
        img_s = transforms.RandomGrayscale(p=0.2)(img_s)
        img_s = blur(img_s, p=0.5)

        if self.mode == "train_l":
            img_w, mask = normalize(img_w, mask)
            img_s = normalize(img_s)
            return img_w, img_s, mask

        dummy_mask = torch.from_numpy(np.array(mask)).long()
        return normalize(img_w), normalize(img_s), dummy_mask

    def __len__(self):
        if self.mode == "train_u":
            return len(self.ids) * 50
        return len(self.ids)
