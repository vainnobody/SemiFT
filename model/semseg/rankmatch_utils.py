"""
RankMatch: Exploring the Better Consistency Regularization
for Semi-supervised Semantic Segmentation

Core utility functions for rank-based consistency regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import permutations


def prob2rank(prob: torch.Tensor, prob_s: torch.Tensor, k: int = 4):
    """
    Convert probability distribution to rank distribution for consistency regularization.

    Args:
        prob: Probability tensor [B, H, W, N] from weak augmentation
        prob_s: Probability tensor [B, H, W, N] from strong augmentation
        k: Top-k classes to consider for ranking (default: 4)

    Returns:
        rank: Rank distribution for weak augmentation [B, H, W, k!]
        rank_s: Rank distribution for strong augmentation [B, H, W, k!]
    """
    full_permutation = [c for c in permutations(range(k))]
    full_permutation = torch.from_numpy(np.stack(full_permutation)).to(
        prob.device
    )  # [k!, k]

    _, prob_topk_index = prob.topk(k, dim=-1)  # [B, H, W, k]
    A = prob_topk_index[:, :, :, full_permutation]  # [B, H, W, k!, k]
    B = prob.unsqueeze(3).expand(
        -1, -1, -1, full_permutation.shape[0], -1
    )  # [B, H, W, k!, N]
    B_s = prob_s.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1)
    C = torch.gather(input=B, dim=-1, index=A)  # [B, H, W, k!, k]
    C_s = torch.gather(input=B_s, dim=-1, index=A)

    rank = C[:, :, :, :, 0] / (C[:, :, :, :, 0:].sum(dim=-1) + 1e-10)  # [B, H, W, k!]
    rank_s = C_s[:, :, :, :, 0] / (C_s[:, :, :, :, 0:].sum(dim=-1) + 1e-10)

    for i in range(1, k):
        rank *= C[:, :, :, :, i] / (C[:, :, :, :, i:].sum(dim=-1) + 1e-10)
        rank_s *= C_s[:, :, :, :, i] / (C_s[:, :, :, :, i:].sum(dim=-1) + 1e-10)

    return rank, rank_s


def orthogonal_landmarks(
    q: torch.Tensor,
    q_s: torch.Tensor,
    num_landmarks: int = 64,
    subsample_fraction: float = 1.0,
):
    """
    Construct set of landmarks by recursively selecting new landmarks
    that are maximally orthogonal to the existing set.

    Args:
        q: Feature tensor [B, C, H, W] from weak augmentation
        q_s: Feature tensor [B, C, H, W] from strong augmentation
        num_landmarks: Number of landmarks to select (default: 64)
        subsample_fraction: Fraction of queries to subsample (default: 1.0)

    Returns:
        landmarks: Selected landmarks [B, M, D]
        landmarks_s: Corresponding landmarks from strong augmentation [B, M, D]
    """
    B, D, H, W = q.shape
    N = H * W
    q = q.permute(0, 2, 3, 1).reshape(B, -1, D)
    q_s = q_s.permute(0, 2, 3, 1).reshape(B, -1, D)

    if subsample_fraction < 1.0:
        # Need at least M/2 samples of queries and keys
        num_samples = max(int(subsample_fraction * q.size(-2)), num_landmarks)
        q_unnormalised = q[
            :, torch.randint(q.size(-2), (num_samples,), device=q.device), :
        ]
    else:
        q_unnormalised = q

    # Normalize for cosine similarity computation
    qk = F.normalize(q_unnormalised, p=2, dim=-1)

    selected_mask = torch.zeros((B, N, 1), device=qk.device)
    landmark_mask = torch.ones((B, 1, 1), dtype=selected_mask.dtype, device=qk.device)

    # Get initial random landmark
    random_idx = torch.randint(qk.size(-2), (B, 1, 1), device=qk.device)
    selected_landmark = qk[torch.arange(qk.size(0)), random_idx.view(-1), :].view(B, D)
    selected_mask.scatter_(-2, random_idx, landmark_mask)

    # Selected landmarks
    selected_landmarks = torch.empty(
        (B, num_landmarks, D), device=qk.device, dtype=qk.dtype
    )
    selected_landmarks[:, 0, :] = selected_landmark

    # Store computed cosine similarities
    cos_sims = torch.empty((B, N, num_landmarks), device=qk.device, dtype=qk.dtype)

    for M in range(1, num_landmarks):
        # Calculate absolute cosine similarity between selected and unselected landmarks
        cos_sim = torch.einsum("b n d, b d -> b n", qk, selected_landmark).abs()
        cos_sims[:, :, M - 1] = cos_sim
        cos_sim_set = cos_sims[:, :, :M]

        # Get orthogonal landmark: landmark with smallest absolute cosine similarity
        cos_sim_set.view(-1, M)[selected_mask.flatten().bool(), :] = 10
        selected_landmark_idx = cos_sim_set.amax(-1).argmin(-1)
        selected_landmark = qk[torch.arange(qk.size(0)), selected_landmark_idx, :].view(
            B, D
        )

        # Add most orthogonal landmark to selected landmarks
        selected_landmarks[:, M, :] = selected_landmark

        # Remove selected indices from non-selected mask
        selected_mask.scatter_(
            -2, selected_landmark_idx.unsqueeze(-1).unsqueeze(-1), landmark_mask
        )

    landmarks = torch.masked_select(q_unnormalised, selected_mask.bool()).reshape(
        B, -1, D
    )
    landmarks_s = torch.masked_select(q_s, selected_mask.bool()).reshape(B, -1, D)

    return landmarks, landmarks_s


def corr_loss(
    feat_w: torch.Tensor,
    feat_s: torch.Tensor,
    local_rank: int,
    num_landmarks: int = 64,
    k: int = 4,
):
    """
    Compute pixel-reference correlation consistency loss.

    Uses orthogonal landmarks as references and computes KL divergence
    between rank distributions of weak and strong augmentation features.

    Args:
        feat_w: Feature tensor [B, C, H, W] from weak augmentation
        feat_s: Feature tensor [B, C, H, W] from strong augmentation
        local_rank: CUDA device rank for KLDivLoss
        num_landmarks: Number of landmarks for correlation (default: 64)
        k: Top-k for ranking (default: 4)

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
