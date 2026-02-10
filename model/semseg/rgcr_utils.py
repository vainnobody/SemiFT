"""
RGCR: Rotation-Geometric Consistency Regularization
for Semi-supervised Semantic Segmentation

Core utility functions combining RVS geometric transforms with
rank-based correlation consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import permutations
import torchvision.transforms.functional as TF


# ========================
# Geometric Transform Utils (from fixmatch_rvsc)
# ========================


def scale_back(pred_c_back, mask_c_back, size, box):
    """
    Scale back rotated/scaled predictions to original coordinate space.

    Args:
        pred_c_back: Predictions from context view [B, C, h, w]
        mask_c_back: Mask from context view [B, 1, h, w]
        size: Crop size
        box: Box parameters [x, y, x_c, y_c, s, theta]

    Returns:
        preds: Aligned predictions [B, C, H, W]
        masks: Valid masks [B, 1, H, W]
    """
    B, C, h, w = pred_c_back.shape

    preds = []
    masks = []

    for i in range(B):
        x, y, x_c, y_c, s, theta = box[i]
        x, y, x_c, y_c, s, theta = (
            x.item(),
            y.item(),
            x_c.item(),
            y_c.item(),
            s.item(),
            theta.item(),
        )

        pred = TF.rotate(
            pred_c_back[i], angle=-theta, interpolation=TF.InterpolationMode.BILINEAR
        )
        mask = TF.rotate(
            mask_c_back[i],
            angle=-theta,
            interpolation=TF.InterpolationMode.NEAREST,
            fill=0,
        )
        aligned_ctx = torch.zeros((C, h, w)).to(pred_c_back.device)
        aligned_mask = torch.zeros((1, h, w)).to(pred_c_back.device)
        rect_m = [x, y, x + size, y + size]
        rect_s = [x_c, y_c, x_c + size * s, y_c + size * s]

        inter_x1 = max(rect_m[0], rect_s[0])
        inter_y1 = max(rect_m[1], rect_s[1])
        inter_x2 = min(rect_m[2], rect_s[2])
        inter_y2 = min(rect_m[3], rect_s[3])

        m_x1 = int(round(inter_x1 - x))
        m_y1 = int(round(inter_y1 - y))
        m_x2 = int(round(inter_x2 - x))
        m_y2 = int(round(inter_y2 - y))

        target_h = m_y2 - m_y1
        target_w = m_x2 - m_x1

        c_x1 = int(round((inter_x1 - x_c) / s))
        c_y1 = int(round((inter_y1 - y_c) / s))
        c_x2 = int(round((inter_x2 - x_c) / s))
        c_y2 = int(round((inter_y2 - y_c) / s))

        c_x1, c_x2 = max(0, c_x1), min(size, c_x2)
        c_y1, c_y2 = max(0, c_y1), min(size, c_y2)

        pred_patch = pred[:, c_y1:c_y2, c_x1:c_x2]
        mask_patch = mask[:, c_y1:c_y2, c_x1:c_x2]

        if (
            target_h > 0
            and target_w > 0
            and pred_patch.shape[1] > 0
            and pred_patch.shape[2] > 0
        ):
            pred_patch = F.interpolate(
                pred_patch.unsqueeze(0), size=(target_h, target_w), mode="bilinear"
            ).squeeze(0)
            mask_patch = F.interpolate(
                mask_patch.unsqueeze(0), size=(target_h, target_w), mode="nearest"
            ).squeeze(0)

            aligned_ctx[:, m_y1:m_y2, m_x1:m_x2] = pred_patch
            aligned_mask[:, m_y1:m_y2, m_x1:m_x2] = mask_patch

        preds.append(aligned_ctx)
        masks.append(aligned_mask)

    return torch.stack(preds, dim=0), torch.stack(masks, dim=0)


def scale_back_features(feat_rvs, mask_c, crop_size, box):
    """
    Scale back RVS-transformed features to original coordinate space.
    Same logic as scale_back but applied to feature maps.

    Args:
        feat_rvs: Feature maps from RVS view [B, C, h, w]
        mask_c: Valid mask [B, 1, H, W] (in rotated space)
        crop_size: Original crop size
        box: Box parameters [x, y, x_c, y_c, s, theta]

    Returns:
        feat_recovered: Recovered features [B, C, H, W]
        valid_masks: Valid region masks [B, 1, H, W]
    """
    return scale_back(feat_rvs, mask_c, crop_size, box)


# ========================
# RankMatch Core Utils (from rankmatch_utils.py)
# ========================


def prob2rank(prob, prob_s, k=4):
    """
    Convert probability distribution to rank distribution.

    Args:
        prob: Probability tensor [B, H, W, N] from weak augmentation
        prob_s: Probability tensor [B, H, W, N] from strong augmentation
        k: Top-k classes to consider for ranking

    Returns:
        rank: Rank distribution [B, H, W, k!]
        rank_s: Rank distribution [B, H, W, k!]
    """
    full_permutation = [c for c in permutations(range(k))]
    full_permutation = torch.from_numpy(np.stack(full_permutation)).to(prob.device)

    _, prob_topk_index = prob.topk(k, dim=-1)
    A = prob_topk_index[:, :, :, full_permutation]
    B = prob.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1)
    B_s = prob_s.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1)
    C = torch.gather(input=B, dim=-1, index=A)
    C_s = torch.gather(input=B_s, dim=-1, index=A)

    rank = C[:, :, :, :, 0] / (C[:, :, :, :, 0:].sum(dim=-1) + 1e-10)
    rank_s = C_s[:, :, :, :, 0] / (C_s[:, :, :, :, 0:].sum(dim=-1) + 1e-10)

    for i in range(1, k):
        rank *= C[:, :, :, :, i] / (C[:, :, :, :, i:].sum(dim=-1) + 1e-10)
        rank_s *= C_s[:, :, :, :, i] / (C_s[:, :, :, :, i:].sum(dim=-1) + 1e-10)

    return rank, rank_s


def orthogonal_landmarks(q, q_s, num_landmarks=64, subsample_fraction=1.0):
    """
    Construct set of landmarks by recursively selecting maximally
    orthogonal features.

    Args:
        q: Feature tensor [B, C, H, W] from weak augmentation
        q_s: Feature tensor [B, C, H, W] from strong augmentation
        num_landmarks: Number of landmarks to select

    Returns:
        landmarks: Selected landmarks [B, M, D]
        landmarks_s: Corresponding landmarks from strong aug [B, M, D]
    """
    B, D, H, W = q.shape
    N = H * W
    q = q.permute(0, 2, 3, 1).reshape(B, -1, D)
    q_s = q_s.permute(0, 2, 3, 1).reshape(B, -1, D)

    if subsample_fraction < 1.0:
        num_samples = max(int(subsample_fraction * q.size(-2)), num_landmarks)
        q_unnormalised = q[
            :, torch.randint(q.size(-2), (num_samples,), device=q.device), :
        ]
    else:
        q_unnormalised = q

    qk = F.normalize(q_unnormalised, p=2, dim=-1)

    selected_mask = torch.zeros((B, N, 1), device=qk.device)
    landmark_mask = torch.ones((B, 1, 1), dtype=selected_mask.dtype, device=qk.device)

    random_idx = torch.randint(qk.size(-2), (B, 1, 1), device=qk.device)
    selected_landmark = qk[torch.arange(qk.size(0)), random_idx.view(-1), :].view(B, D)
    selected_mask.scatter_(-2, random_idx, landmark_mask)

    selected_landmarks = torch.empty(
        (B, num_landmarks, D), device=qk.device, dtype=qk.dtype
    )
    selected_landmarks[:, 0, :] = selected_landmark

    cos_sims = torch.empty((B, N, num_landmarks), device=qk.device, dtype=qk.dtype)

    for M in range(1, num_landmarks):
        cos_sim = torch.einsum("b n d, b d -> b n", qk, selected_landmark).abs()
        cos_sims[:, :, M - 1] = cos_sim
        cos_sim_set = cos_sims[:, :, :M]

        cos_sim_set.view(-1, M)[selected_mask.flatten().bool(), :] = 10
        selected_landmark_idx = cos_sim_set.amax(-1).argmin(-1)
        selected_landmark = qk[torch.arange(qk.size(0)), selected_landmark_idx, :].view(
            B, D
        )

        selected_landmarks[:, M, :] = selected_landmark

        selected_mask.scatter_(
            -2, selected_landmark_idx.unsqueeze(-1).unsqueeze(-1), landmark_mask
        )

    landmarks = torch.masked_select(q_unnormalised, selected_mask.bool()).reshape(
        B, -1, D
    )
    landmarks_s = torch.masked_select(q_s, selected_mask.bool()).reshape(B, -1, D)

    return landmarks, landmarks_s


# ========================
# RGCR-specific Loss Functions
# ========================


def corr_loss(feat_w, feat_s, local_rank, num_landmarks=64, k=4):
    """
    Compute pixel-reference correlation consistency loss (from RankMatch).

    Args:
        feat_w: Feature tensor [B, C, H, W] from weak augmentation
        feat_s: Feature tensor [B, C, H, W] from strong augmentation
        local_rank: CUDA device rank for KLDivLoss
        num_landmarks: Number of landmarks
        k: Top-k for ranking

    Returns:
        loss: Correlation consistency loss (scalar)
    """
    criterion_c = nn.KLDivLoss(reduction="mean").cuda(local_rank)

    refers_w, refers_s = orthogonal_landmarks(feat_w, feat_s, num_landmarks)

    p2r_w = torch.einsum("b c h w, b n c -> b h w n", feat_w, refers_w).softmax(dim=-1)
    p2r_s = torch.einsum("b c h w, b n c -> b h w n", feat_s, refers_s).softmax(dim=-1)

    p2r_w_rank, p2r_s_rank = prob2rank(p2r_w, p2r_s, k=k)

    loss = criterion_c((p2r_s_rank + 1e-10).log(), p2r_w_rank)

    return loss


def geometric_corr_loss(
    feat_w, feat_rvs_recovered, valid_masks, local_rank, num_landmarks=64, k=4
):
    """
    Compute geometric-aware correlation consistency loss.

    Uses recovered RVS features (aligned to original space) and computes
    rank-based correlation consistency against weak augmentation features.
    Only considers valid regions where the RVS recovery is reliable.

    Args:
        feat_w: Feature tensor [B, C, H, W] from weak augmentation
        feat_rvs_recovered: Recovered RVS features [B, C, H, W] aligned to
                            the same coordinate space as feat_w
        valid_masks: Valid region masks [B, 1, H, W], 1 = valid
        local_rank: CUDA device rank
        num_landmarks: Number of landmarks
        k: Top-k for ranking

    Returns:
        loss: Geometric correlation consistency loss (scalar)
    """
    # Mask out invalid regions (where scale_back couldn't recover)
    # by zeroing features in invalid regions
    feat_w_masked = feat_w * valid_masks
    feat_rvs_masked = feat_rvs_recovered * valid_masks

    # Use the standard corr_loss on the masked features
    loss = corr_loss(feat_w_masked, feat_rvs_masked, local_rank, num_landmarks, k)

    return loss
