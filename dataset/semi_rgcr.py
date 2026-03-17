from copy import deepcopy
import math
import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from dataset.transform import (
    blur,
    context_crop,
    crop,
    crop_with_xy,
    hflip,
    normalize,
    obtain_cutmix_box,
    resize,
)


class SemiDataset(Dataset):
    def __init__(
        self, name, root, mode, size=None, id_path=None, nsample=None, ignore_index=255
    ):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_index = ignore_index

        if mode == "train_l" or mode == "train_u":
            with open(id_path, "r") as f:
                self.ids = f.read().splitlines()
            if mode == "train_l" and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open("splits/%s/val.txt" % name, "r") as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        sample_id = random.choice(self.ids)
        img = Image.open(os.path.join(self.root, sample_id.split(" ")[0])).convert(
            "RGB"
        )
        if self.mode == "train_u":
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask = Image.fromarray(
                np.array(Image.open(os.path.join(self.root, sample_id.split(" ")[1])))
            )

        if self.mode == "val":
            img, mask = normalize(img, mask)
            return img, mask, sample_id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == "train_u" else self.ignore_index

        if self.mode == "train_l":
            img, mask = crop(img, mask, self.size, ignore_value)
            img, mask = hflip(img, mask, p=0.5)
            return normalize(img, mask)

        img_crop, mask_crop, x, y = crop_with_xy(img, mask, self.size, ignore_value)
        img_w, img_s1, img_s2 = (
            deepcopy(img_crop),
            deepcopy(img_crop),
            deepcopy(img_crop),
        )

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        theta = np.random.uniform(0.0, 360.0)
        scale = float(np.random.choice([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]))
        img_u_rvs, _, x_c, y_c = context_crop(
            img, mask, self.size, ignore_value, x, y, scale
        )
        img_u_rvs = img_u_rvs.resize(img_crop.size)
        img_u_rvs = TF.rotate(
            img_u_rvs,
            angle=theta,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        ignore_mask = Image.fromarray(
            np.zeros((mask_crop.size[1], mask_crop.size[0]), dtype=np.uint8)
        )
        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask_crop = torch.from_numpy(np.array(mask_crop)).long()
        ignore_mask[mask_crop == ignore_value] = 255

        mask_c = torch.ones((1, self.size, self.size))
        mask_rotated = TF.rotate(
            mask_c,
            angle=theta,
            interpolation=TF.InterpolationMode.NEAREST,
            fill=0,
        )

        return (
            normalize(img_w),
            img_s1,
            img_s2,
            normalize(img_u_rvs),
            ignore_mask,
            cutmix_box1,
            cutmix_box2,
            torch.tensor([x, y, x_c, y_c, scale, theta], dtype=torch.float32),
            mask_rotated,
        )

    def __len__(self):
        return len(self.ids) * 50
