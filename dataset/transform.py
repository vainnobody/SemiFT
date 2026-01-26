import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import math
import torch
from torchvision import transforms


def context_crop(img, mask, size, ignore_value, x, y, s):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0,0,padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0,0,padw, padh), fill=ignore_value)
    
    w, h = img.size
    box_w, box_h = s * size, s * size
    if s > 1.0:
        delta_s = np.random.uniform(0, s - 1.0)
        cur_x, cur_y = x - delta_s * size, y - delta_s * size
    elif s < 1.0:
        delta_s = np.random.uniform(0, 1.0 - s)
        cur_x, cur_y = x + delta_s * size, y + delta_s * size
    else:
        cur_x, cur_y = x, y
    img = img.crop((cur_x, cur_y, cur_x + box_w, cur_y + box_h))
    mask = mask.crop((cur_x, cur_y, cur_x + box_w, cur_y + box_h))
    return img, mask, cur_x, cur_y


def crop_with_xy(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask, x, y


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask



def crop_overlap(img, mask, size, ignore_value=255):
    """
    对输入的图像和 mask 进行扩充并裁剪出两个 crop，要求两个 crop 有至少50%的连续空间重合，
    同时返回两个大小为 size×size 的重合区域二值矩阵（分别表示在 crop1 和 crop2 中的重合区域）。

    Parameters:
      img (PIL.Image): 输入图像
      mask (PIL.Image): 输入 mask
      size (int): 裁剪区域的边长
      ignore_value (int): mask 扩充时使用的填充值

    Returns:
      crop1_img, crop1_mask: 第一个 crop 的图像和 mask
      crop2_img, crop2_mask: 第二个 crop 的图像和 mask
      overlap1, overlap2: 两个 (size, size) 的二值图，分别表示两个 crop 内的重合区域（值为1）
    """
    # 扩充图像和 mask，确保尺寸不小于 size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)
    w, h = img.size

    # 第一个 crop 的坐标
    x1 = random.randint(0, w - size)
    y1 = random.randint(0, h - size)
    crop1_img = img.crop((x1, y1, x1 + size, y1 + size))
    crop1_mask = mask.crop((x1, y1, x1 + size, y1 + size))

    # 第二个 crop 的坐标，限制偏移保证至少 50% 重合
    max_offset = size - int(math.ceil(size / math.sqrt(2)))
    dx = random.randint(-max_offset, max_offset)
    dy = random.randint(-max_offset, max_offset)
    x2 = max(0, min(x1 + dx, w - size))
    y2 = max(0, min(y1 + dy, h - size))
    crop2_img = img.crop((x2, y2, x2 + size, y2 + size))
    crop2_mask = mask.crop((x2, y2, x2 + size, y2 + size))

    # 交集区域在整张图上的范围
    inter_left = max(x1, x2)
    inter_top = max(y1, y2)
    inter_right = min(x1 + size, x2 + size)
    inter_bottom = min(y1 + size, y2 + size)

    # 在 crop1 和 crop2 的坐标系中分别表示交集位置
    overlap1_array = np.zeros((size, size), dtype=np.uint8)
    overlap2_array = np.zeros((size, size), dtype=np.uint8)

    if inter_right > inter_left and inter_bottom > inter_top:
        # crop1 中的交集区域
        rel_left1 = inter_left - x1
        rel_top1 = inter_top - y1
        rel_right1 = rel_left1 + (inter_right - inter_left)
        rel_bottom1 = rel_top1 + (inter_bottom - inter_top)
        overlap1_array[rel_top1:rel_bottom1, rel_left1:rel_right1] = 1

        # crop2 中的交集区域
        rel_left2 = inter_left - x2
        rel_top2 = inter_top - y2
        rel_right2 = rel_left2 + (inter_right - inter_left)
        rel_bottom2 = rel_top2 + (inter_bottom - inter_top)
        overlap2_array[rel_top2:rel_bottom2, rel_left2:rel_right2] = 1

    overlap1 = Image.fromarray(overlap1_array, mode='L')
    overlap2 = Image.fromarray(overlap2_array, mode='L')
    return crop1_img, crop1_mask, crop2_img, crop2_mask, overlap1, overlap2



def crop_overlap_two(img, mask, size, min_overlap_ratio=0.75, ignore_value=255):
    """
    裁剪三张空间上重叠的图像区域（与 crop1 至少 min_overlap_ratio 重叠），
    并返回它们各自与 crop1 的重叠区域图。

    Args:
        img (PIL.Image)
        mask (PIL.Image)
        size (int): 裁剪的正方形边长
        min_overlap_ratio (float): 与 crop1 最小重叠面积比例 (0~1)
        ignore_value (int): 用于 mask 填充的值

    Returns:
        crop1_img, crop1_mask,
        crop2_img, crop2_mask,
        crop3_img, crop3_mask,
        overlap1_2, overlap2,
        overlap1_3, overlap3
    """
    assert 0 < min_overlap_ratio <= 1.0, "min_overlap_ratio must be in (0, 1]"

    # Step 1: 填充图像和 mask
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    if padw > 0 or padh > 0:
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)
    w, h = img.size

    # Step 2: 随机第一个 crop 的坐标
    x1 = random.randint(0, w - size)
    y1 = random.randint(0, h - size)

    def sample_neighbor(x1, y1):
        # 计算最小重叠区域对应的最大偏移量（以像素为单位）
        min_overlap_size = int(size * math.sqrt(min_overlap_ratio))
        max_offset = size - min_overlap_size
        dx = random.randint(-max_offset, max_offset)
        dy = random.randint(-max_offset, max_offset)
        x = max(0, min(x1 + dx, w - size))
        y = max(0, min(y1 + dy, h - size))
        return x, y

    def get_overlap_mask(xA, yA, xB, yB):
        inter_left = max(xA, xB)
        inter_top = max(yA, yB)
        inter_right = min(xA + size, xB + size)
        inter_bottom = min(yA + size, yB + size)

        maskA = np.zeros((size, size), dtype=np.uint8)
        maskB = np.zeros((size, size), dtype=np.uint8)

        if inter_right > inter_left and inter_bottom > inter_top:
            rel_leftA = inter_left - xA
            rel_topA = inter_top - yA
            rel_rightA = rel_leftA + (inter_right - inter_left)
            rel_bottomA = rel_topA + (inter_bottom - inter_top)
            maskA[rel_topA:rel_bottomA, rel_leftA:rel_rightA] = 1

            rel_leftB = inter_left - xB
            rel_topB = inter_top - yB
            rel_rightB = rel_leftB + (inter_right - inter_left)
            rel_bottomB = rel_topB + (inter_bottom - inter_top)
            maskB[rel_topB:rel_bottomB, rel_leftB:rel_rightB] = 1

        return (
            Image.fromarray(maskA, mode='L'),
            Image.fromarray(maskB, mode='L')
        )

    # Crop 1
    crop1_img = img.crop((x1, y1, x1 + size, y1 + size))
    crop1_mask = mask.crop((x1, y1, x1 + size, y1 + size))

    # Crop 2
    x2, y2 = sample_neighbor(x1, y1)
    crop2_img = img.crop((x2, y2, x2 + size, y2 + size))
    crop2_mask = mask.crop((x2, y2, x2 + size, y2 + size))
    overlap1_2, overlap2 = get_overlap_mask(x1, y1, x2, y2)

    # Crop 3
    x3, y3 = sample_neighbor(x1, y1)
    crop3_img = img.crop((x3, y3, x3 + size, y3 + size))
    crop3_mask = mask.crop((x3, y3, x3 + size, y3 + size))
    overlap1_3, overlap3 = get_overlap_mask(x1, y1, x3, y3)

    return (
        crop1_img, crop1_mask,
        crop2_img, crop2_mask,
        crop3_img, crop3_mask,
        overlap1_2, overlap2,
        overlap1_3, overlap3
    )

def crop_overlap_three(img, mask, size, ignore_value=255):
    """
    裁剪四张空间上重叠的图像区域（每一张都与 crop1 至少50%重叠），
    返回图像、mask 以及它们与 crop1 的重叠区域图（crop1坐标系和当前crop坐标系下的重叠区域）。
    """
    # Step 1: 填充图像和 mask（镜像方式）
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    if padw > 0 or padh > 0:
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)
    w, h = img.size

    # Step 2: 第一个 crop 的坐标
    x1 = random.randint(0, w - size)
    y1 = random.randint(0, h - size)

    def sample_neighbor(x1, y1):
        max_offset = size - int(math.ceil(size / math.sqrt(2)))
        dx = random.randint(-max_offset, max_offset)
        dy = random.randint(-max_offset, max_offset)
        x = max(0, min(x1 + dx, w - size))
        y = max(0, min(y1 + dy, h - size))
        return x, y

    def get_overlap_mask(xA, yA, xB, yB):
        inter_left = max(xA, xB)
        inter_top = max(yA, yB)
        inter_right = min(xA + size, xB + size)
        inter_bottom = min(yA + size, yB + size)

        maskA = np.zeros((size, size), dtype=np.uint8)
        maskB = np.zeros((size, size), dtype=np.uint8)

        if inter_right > inter_left and inter_bottom > inter_top:
            rel_leftA = inter_left - xA
            rel_topA = inter_top - yA
            rel_rightA = rel_leftA + (inter_right - inter_left)
            rel_bottomA = rel_topA + (inter_bottom - inter_top)
            maskA[rel_topA:rel_bottomA, rel_leftA:rel_rightA] = 1

            rel_leftB = inter_left - xB
            rel_topB = inter_top - yB
            rel_rightB = rel_leftB + (inter_right - inter_left)
            rel_bottomB = rel_topB + (inter_bottom - inter_top)
            maskB[rel_topB:rel_bottomB, rel_leftB:rel_rightB] = 1

        return (
            Image.fromarray(maskA, mode='L'),
            Image.fromarray(maskB, mode='L')
        )

    # Crop 1
    crop1_img = img.crop((x1, y1, x1 + size, y1 + size))
    crop1_mask = mask.crop((x1, y1, x1 + size, y1 + size))

    results = [crop1_img, crop1_mask]

    for _ in range(3):  # Crop2, Crop3, Crop4
        xN, yN = sample_neighbor(x1, y1)
        cropN_img = img.crop((xN, yN, xN + size, yN + size))
        cropN_mask = mask.crop((xN, yN, xN + size, yN + size))
        overlap1N, overlapN = get_overlap_mask(x1, y1, xN, yN)
        results.extend([cropN_img, cropN_mask, overlap1N, overlapN])

    return tuple(results)


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
