import numpy as np
import torch


def generate_cutmix_mask(img_size, ratio=2):
    cut_area = img_size[0] * img_size[1] / ratio
    w = np.random.randint(int(img_size[1] / ratio) + 1, img_size[1] + 1)
    h = int(np.round(cut_area / w))
    h = max(1, min(h, img_size[0]))
    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)
    x_end = int(x_start + w)
    y_end = int(y_start + h)
    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.long()


def generate_unsup_aug_sc(conf_w, mask_w, data_s):
    b, _, h, w = data_s.shape
    device = data_s.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([h, w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_data_s.append((data_s[i] * augmix_mask + data_s[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    return torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s)


def generate_unsup_aug_ds(data_s1, data_s2):
    b, _, h, w = data_s1.shape
    device = data_s1.device
    new_data_s = []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([h, w]).to(device)
        new_data_s.append((data_s1[i] * augmix_mask + data_s2[i] * (1 - augmix_mask)).unsqueeze(0))
    return torch.cat(new_data_s)


def generate_unsup_aug_dc(conf_w, mask_w, data_s1, data_s2):
    b, _, h, w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([h, w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_data_s.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    return torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s)


def generate_unsup_aug_sdc(conf_w, mask_w, data_s1, data_s2):
    b, _, h, w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([h, w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        if i % 2 == 0:
            mixed = data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)
        else:
            mixed = data_s2[i] * augmix_mask + data_s1[(i + 1) % b] * (1 - augmix_mask)
        new_data_s.append(mixed.unsqueeze(0))
    return torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s)


def entropy_map(probs, dim=1):
    return -torch.sum(probs * torch.log2(probs + 1e-10), dim=dim)
