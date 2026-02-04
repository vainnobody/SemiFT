import os
import math
import random
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.transform import *
import torchvision.transforms.functional as TF


class SemiDataset(Dataset):
    def __init__(
        self, name, root, mode, size=None, ignore_value=5, id_path=None, nsample=None
    ):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.reduce_zero_label = True if name == "ade20k" else False

        if mode == "train_l" or mode == "train_u":
            with open(id_path, "r") as f:
                self.ids = f.read().splitlines()
            if mode == "train_l" and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open("splits/%s/val.txt" % name, "r") as f:
                self.ids = f.read().splitlines()

    def process_mask(self, mask):
        mask = np.array(mask) - 1
        return Image.fromarray(mask)

    def __getitem__(self, item):
        id = random.choice(self.ids)
        img_ori = Image.open(os.path.join(self.root, id.split(" ")[0])).convert("RGB")
        #  mask_ori = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == "train_u":
            mask_ori = Image.fromarray(
                np.zeros((img_ori.size[1], img_ori.size[0]), dtype=np.uint8)
            )
        else:
            mask_ori = Image.fromarray(
                np.array(Image.open(os.path.join(self.root, id.split(" ")[1])))
            )
            if self.name == "loveda":
                mask_ori = self.process_mask(mask_ori)

        ignore_value = 254 if self.mode == "train_u" else 5

        img, mask, x, y = crop_with_xy(img_ori, mask_ori, self.size, ignore_value)

        min_theta = 0.0
        max_theta = 360.0
        theta = np.random.uniform(min_theta, max_theta)
        # theta = np.random.choice([90.0, 180.0, 270.0])

        s = float(np.random.choice([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]))

        img_c, mask_c, x_c, y_c = context_crop(
            img_ori, mask_ori, self.size, ignore_value, x, y, s
        )

        img_c = img_c.resize(img.size)
        mask_c = mask_c.resize(img.size, resample=Image.Resampling.NEAREST)

        if self.mode == "train_l":
            img_c = TF.rotate(
                img_c, angle=theta, interpolation=TF.InterpolationMode.BILINEAR
            )
            mask_c = TF.rotate(
                mask_c, angle=theta, interpolation=TF.InterpolationMode.NEAREST, fill=0
            )
            img, mask = normalize(img, mask)
            img_c, mask_c = normalize(img_c, mask_c)
            return img, mask, img_c, mask_c

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img_c)

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

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        mask_c = torch.ones((1, self.size, self.size))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        img_s2 = TF.rotate(
            img_s1, angle=theta, interpolation=TF.InterpolationMode.BILINEAR
        )
        mask_rotated = TF.rotate(
            mask_c, angle=theta, interpolation=TF.InterpolationMode.NEAREST, fill=0
        )

        return (
            normalize(img_w),
            img_s1,
            img_s2,
            ignore_mask,
            cutmix_box1,
            cutmix_box2,
            torch.tensor([x, y, x_c, y_c, s, theta]),
            mask_rotated,
        )

    def __len__(self):
        # return 1000
        return len(self.ids) * 50
