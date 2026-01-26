import os
import math
import random
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from dataset.transform import *


class ValDataset(Dataset):
    def __init__(self, name, root, mode, size=None,  ignore_value=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.reduce_zero_label = True if name == 'ade20k' else False

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        # print("len(self.ids) - 1", len(self.ids) - 1)
        # print("self.ids", self.ids)
        # id = self.ids[len(self.ids) - 1]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask =Image.open(os.path.join(self.root, id.split(' ')[1]))

        if self.reduce_zero_label:
            mask = np.array(mask)
            mask[mask == 0] = 255
            mask = mask - 1
            mask[mask == 254] = 255
            mask = Image.fromarray(mask)

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

    def __len__(self):
        return len(self.ids)
