"""
SegMind utility functions for semi-supervised semantic segmentation.
Includes mask generation, ClassMix augmentation, and contrastive learning loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Mask Generation Functions
# ============================================================================


def get_mask_tensor(h=512, w=512, mask_gap=16, mask_rate=0.75):
    """
    Generate a random mask tensor for masked image modeling.

    Args:
        h: Image height
        w: Image width
        mask_gap: Size of each mask block
        mask_rate: Ratio of masked pixels (0.0 to 1.0)

    Returns:
        mask_tensor: [1, h, w] tensor where 0=masked, 1=visible
    """
    # Use ceiling division to handle non-divisible dimensions
    h_gap_num = (h + mask_gap - 1) // mask_gap
    w_gap_num = (w + mask_gap - 1) // mask_gap

    mask_tensor_small = (
        torch.randperm(h_gap_num * w_gap_num).float().reshape((h_gap_num, w_gap_num))
    )
    divide_threshold = h_gap_num * w_gap_num * mask_rate
    mask_tensor_small[mask_tensor_small < divide_threshold] = 0.0
    mask_tensor_small[mask_tensor_small >= divide_threshold] = 1.0

    # Interpolate to full size and crop to exact dimensions
    mask_tensor = F.interpolate(
        mask_tensor_small.unsqueeze(0).unsqueeze(0),
        size=(h_gap_num * mask_gap, w_gap_num * mask_gap),
        mode="nearest",
    )
    # Crop to exact h, w
    mask_tensor = mask_tensor[:, :, :h, :w].squeeze(0)

    return mask_tensor


def get_batch_mask_tensor(nchw=(1, 3, 512, 512), mask_gap=16, mask_rate=0.75):
    """
    Generate batch of random mask tensors.

    Args:
        nchw: Tuple of (batch_size, channels, height, width)
        mask_gap: Size of each mask block
        mask_rate: Ratio of masked pixels

    Returns:
        mask_tensor: [N, 1, H, W] tensor
    """
    mask_tensor = torch.zeros((nchw[0], nchw[-2], nchw[-1]))
    for img_i in range(nchw[0]):
        mask_tensor[img_i] = get_mask_tensor(
            h=nchw[-2], w=nchw[-1], mask_gap=mask_gap, mask_rate=mask_rate
        )
    return mask_tensor.unsqueeze(1)


# ============================================================================
# ClassMix Data Augmentation
# ============================================================================


def generate_class_mask(pseudo_labels):
    """
    Generate a class-based mask for ClassMix augmentation.

    Args:
        pseudo_labels: Pseudo label tensor [H, W]

    Returns:
        mask: Binary mask [H, W] where 1 indicates selected classes
    """
    labels = torch.unique(pseudo_labels)
    labels_select = labels[torch.randperm(len(labels))][: len(labels) // 2]

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()


def generate_u_data(img_u_w, img_u_s, lab_u, logits_u, entropy_u, device):
    """
    Generate ClassMix augmented unlabeled data.

    Args:
        img_u_w: Weakly augmented unlabeled images [B, 3, H, W]
        img_u_s: Strongly augmented unlabeled images [B, 3, H, W]
        lab_u: Pseudo labels [B, H, W]
        logits_u: Pseudo label confidence logits [B, H, W]
        entropy_u: Entropy map [B, H, W]
        device: Target device

    Returns:
        Tuple of ClassMix augmented (img_w, img_s, lab, logits, entropy)
    """
    batch_size, _, im_h, im_w = img_u_w.shape

    new_img_w = []
    new_img_s = []
    new_lab = []
    new_logits = []
    new_entropy = []

    for i in range(batch_size):
        mix_mask = generate_class_mask(lab_u[i]).to(device)

        new_img_w.append(
            (
                img_u_w[i] * mix_mask + img_u_w[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_img_s.append(
            (
                img_u_s[i] * mix_mask + img_u_s[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_lab.append(
            (
                lab_u[i] * mix_mask + lab_u[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_logits.append(
            (
                logits_u[i] * mix_mask + logits_u[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_entropy.append(
            (
                entropy_u[i] * mix_mask
                + entropy_u[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )

    new_img_w, new_img_s, new_lab, new_logits, entropy_u = (
        torch.cat(new_img_w),
        torch.cat(new_img_s),
        torch.cat(new_lab),
        torch.cat(new_logits),
        torch.cat(new_entropy),
    )
    return new_img_w, new_img_s, new_lab.long(), new_logits, entropy_u


# ============================================================================
# Contrastive Learning Loss with Memory Bank
# ============================================================================


@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    """
    Update memory bank queue with new keys.

    Args:
        keys: New feature keys to add
        queue: Memory bank list
        queue_ptr: Queue pointer
        queue_size: Maximum queue size
    """
    keys_num = keys.shape[0]
    ptr = int(queue_ptr)
    queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + keys_num) % queue_size
    queue_ptr[0] = ptr


def get_negative_feat(samp_num, memo_list, num_query, num_negative, feat_num):
    """
    Sample negative features from memory bank.

    Args:
        samp_num: Number of samples per class [num_query, num_classes]
        memo_list: Memory bank list
        num_query: Number of queries
        num_negative: Number of negatives per query
        feat_num: Feature dimension

    Returns:
        negative_feat_all: Negative features [num_query, num_negative, feat_num]
    """
    negative_feat_all = torch.zeros((num_query, num_negative, feat_num))
    for i in range(samp_num.shape[0]):
        negative_feat_i_list = []
        for j in range(samp_num.shape[1]):
            if memo_list[j][0].shape[0] == 0:
                continue
            negative_index = np.random.randint(
                low=0, high=memo_list[j][0].shape[0], size=int(samp_num[i, j])
            ).tolist()
            negative_feat_i_list.append(memo_list[j][0][negative_index])
        if len(negative_feat_i_list) > 0:
            negative_feat_i = torch.cat(negative_feat_i_list)
            negative_num = negative_feat_i.shape[0]
            negative_feat_all[i, :negative_num] = negative_feat_i
    return negative_feat_all


def cal_c_loss(
    feat,
    lab,
    prob,
    class_num,
    memory_bank_list,
    queue_size,
    queue_ptr_list,
    query_threshold=0.97,
    temperature=0.5,
    num_query=256,
    num_negative=512,
    device="cuda",
):
    """
    Calculate contrastive learning loss with memory bank.

    Args:
        feat: Feature tensor [B, C, H, W]
        lab: Label tensor [B, H, W]
        prob: Probability tensor [B, num_classes, H, W]
        class_num: Number of classes
        memory_bank_list: List of memory banks per class
        queue_size: Size of each memory bank
        queue_ptr_list: List of queue pointers
        query_threshold: Threshold for hard sample selection
        temperature: Temperature for contrastive loss
        num_query: Number of query samples
        num_negative: Number of negative samples
        device: Target device

    Returns:
        loss_c: Contrastive loss
    """
    device = torch.device(device)
    loss_c = torch.tensor(0.0).to(device)
    feat_num = feat.shape[1]
    feat = feat.permute(0, 2, 3, 1)  # B*C*H*W -> B*H*W*C

    valid_class_batch_list = []
    feat_mean_batch_list = []
    feat_hard_batch_list = []
    feat_mean_set_tensor = torch.zeros((class_num, feat_num))  # c*C

    for class_i in range(class_num):
        lab_i_place = lab == class_i  # B*H*W
        if torch.sum(lab_i_place) == 0:
            continue
        valid_class_batch_list.append(class_i)

        prob_i = prob[:, class_i, :, :]  # B*H*W
        feat_i_hard_mask = (prob_i < query_threshold) * lab_i_place  # B*H*W

        dequeue_and_enqueue(
            keys=feat[lab_i_place],
            queue=memory_bank_list[class_i],
            queue_ptr=queue_ptr_list[class_i],
            queue_size=queue_size[class_i],
        )
        feat_mean_batch_list.append(
            torch.mean(feat[lab_i_place], dim=0, keepdim=True)
        )  # 1*C
        feat_hard_batch_list.append(feat[feat_i_hard_mask])  # hnum*C
        if len(memory_bank_list[class_i][0]) > 0:
            feat_mean_set_tensor[class_i] = torch.mean(
                memory_bank_list[class_i][0], dim=0
            )

    valid_class_num = len(valid_class_batch_list)
    if valid_class_num == 0:
        return loss_c

    for v_class_i in range(valid_class_num):
        v_class_kind = valid_class_batch_list[v_class_i]
        if len(feat_hard_batch_list[v_class_i]) > 0:
            feat_hard_idx = torch.randint(
                len(feat_hard_batch_list[v_class_i]), size=(num_query,)
            )
            query_feat = feat_hard_batch_list[v_class_i][feat_hard_idx]
        else:
            continue

        with torch.no_grad():
            feat_mean_sim = torch.cosine_similarity(
                feat_mean_batch_list[v_class_i], feat_mean_set_tensor.to(device), dim=1
            )
            feat_mean_sim = torch.cat(
                (feat_mean_sim[:v_class_kind], feat_mean_sim[v_class_kind + 1 :])
            )
            negative_sample_prob = torch.softmax(feat_mean_sim, dim=0)
            negative_sample_prob = torch.tensor(
                negative_sample_prob[:v_class_kind].tolist()
                + [0]
                + negative_sample_prob[v_class_kind:].tolist()
            )
            negative_num_sampler = torch.distributions.categorical.Categorical(
                probs=negative_sample_prob
            )
            sample_class = negative_num_sampler.sample(
                sample_shape=[num_query, num_negative]
            )
            sample_class_num = torch.stack(
                [(sample_class == c).sum(1) for c in range(class_num)], dim=1
            )

            negative_feat = get_negative_feat(
                sample_class_num, memory_bank_list, num_query, num_negative, feat_num
            ).to(device)
            positive_feat = (
                feat_mean_batch_list[v_class_i].unsqueeze(0).repeat(num_query, 1, 1)
            )
            all_feat = torch.cat((positive_feat, negative_feat), dim=1)

        seg_logits = torch.cosine_similarity(
            query_feat.unsqueeze(1), all_feat, dim=2
        ).to(device)
        loss_c += F.cross_entropy(
            seg_logits / temperature, torch.zeros(num_query).long().to(device)
        )

    return loss_c / valid_class_num


# ============================================================================
# Memory Bank Initialization
# ============================================================================


def init_memory_bank(class_num, bank_size=10000, feat_dim=256):
    """
    Initialize memory bank for contrastive learning.

    Args:
        class_num: Number of classes
        bank_size: Size of memory bank per class
        feat_dim: Feature dimension

    Returns:
        Tuple of (memory_bank_list, queue_size, queue_ptr_list)
    """
    memory_bank_list = []
    queue_size = []
    queue_ptr_list = []

    for i in range(class_num):
        memory_bank_list.append([torch.zeros(0, feat_dim)])
        queue_size.append(bank_size)
        queue_ptr_list.append(torch.zeros(1, dtype=torch.long))

    return memory_bank_list, queue_size, queue_ptr_list
