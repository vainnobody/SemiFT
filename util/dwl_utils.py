"""
DWL (Distribution-aware Weighting) Utilities for Semi-Supervised Learning

Ported from RS-DWL implementation to SemiFT/FixMatch style.
"""

import torch
import torch.nn.functional as F


def move_cls_memory_to_device(cls_memory: dict, device: torch.device) -> dict:
    """Move all class-memory tensors to the target device in-place."""
    for cls_idx, tensors in cls_memory.items():
        cls_memory[cls_idx] = [tensor.to(device) for tensor in tensors]
    return cls_memory


def update_cls_memory(
    cls_memory: dict,
    pred: torch.Tensor,
    label: torch.Tensor,
    memory_n_batches: int = 50,
) -> dict:
    """
    Update class-wise memory banks with prediction probabilities.

    Args:
        cls_memory: Dictionary mapping class index to list of probability tensors
        pred: Prediction probabilities, shape (N, num_classes)
        label: Labels for each pixel/sample, shape (N,)
        memory_n_batches: Maximum number of batches to keep in memory

    Returns:
        Updated cls_memory dictionary
    """
    N, num_classes = pred.size()

    for i in range(num_classes):
        if (label == i).sum() > 0:
            pred_i = pred[label == i]
            if len(cls_memory[i]) < memory_n_batches:
                cls_memory[i].append(pred_i)
            else:
                cls_memory[i].pop(0)
                cls_memory[i].append(pred_i)

    return cls_memory


def sample_cls_bins(
    cls_memory: dict, num_bins: int = 20, softmax: bool = False
) -> torch.Tensor:
    """
    Sample quantile bins from class-wise memory for distribution-aware weighting.

    Args:
        cls_memory: Dictionary mapping class index to list of probability tensors
        num_bins: Number of quantile bins to sample
        softmax: Whether to apply softmax to the sampled bins

    Returns:
        Tensor of shape (num_classes, num_bins) containing quantile values
    """
    num_classes = len(cls_memory)
    first_tensor = None
    for tensors in cls_memory.values():
        if len(tensors) > 0:
            first_tensor = tensors[0]
            break
    if first_tensor is None:
        raise ValueError("cls_memory must contain at least one tensor per class")

    device = first_tensor.device
    cls_bins = torch.ones(num_classes, num_bins, device=device)

    for i in range(num_classes):
        if len(cls_memory[i]) > 0:
            cls_memory_i = torch.cat([tensor.to(device) for tensor in cls_memory[i]], dim=0)
            # Get the confidence for class i (diagonal of the softmax)
            cls_memory_i = cls_memory_i[:, i]

            sorted_memory_i, _ = torch.sort(cls_memory_i)
            sampled_indices = (
                torch.linspace(0, len(sorted_memory_i) - 1, num_bins + 1).round().long()
            )
            sorted_memory_i = sorted_memory_i[sampled_indices[1:]]

            if softmax:
                cls_bins[i] = F.softmax(sorted_memory_i, dim=-1)
            else:
                cls_bins[i] = sorted_memory_i

    return cls_bins


def calc_wgt_bins(
    cls_bins: torch.Tensor,
    probs: torch.Tensor,
    labels: torch.Tensor,
    iters: int,
    total_iters: int,
) -> torch.Tensor:
    """
    Calculate distribution-aware weights for each sample based on its position in class distribution.

    Uses a sigmoid scaling function that becomes sharper as training progresses.

    Args:
        cls_bins: Quantile bins for each class, shape (num_classes, num_bins)
        probs: Confidence probabilities for each sample, shape (N,)
        labels: Predicted labels for each sample, shape (N,)
        iters: Current iteration number
        total_iters: Total number of iterations

    Returns:
        Weights for each sample, shape (N,), in range [-1, 1] mapped through sigmoid
    """

    def custom_function(x, k):
        """Sigmoid-like function that maps [0, 1] to [-1, 1]"""
        return 1 / (1 + torch.exp(-k * x)) * 2 - 1

    num_bins = cls_bins.size(1)
    cls_exts = cls_bins[labels]  # N x B
    probs_expanded = probs.unsqueeze(dim=-1)  # N x 1

    # Count how many bins the probability exceeds, normalized to [0, 1]
    wgts = ((probs_expanded > cls_exts).sum(dim=-1)).clamp(0, num_bins) / num_bins  # N

    # Progressive sigmoid sharpening: k increases from 2 to 22 over training
    k_parameter = iters / total_iters * 20 + 2
    wgts = custom_function(wgts, k_parameter)

    return wgts


def init_cls_memory(num_classes: int, device: torch.device = None) -> dict:
    """
    Initialize empty class-wise memory banks.

    Args:
        num_classes: Number of classes
        device: Device for tensors (default: cuda if available)

    Returns:
        Dictionary mapping class index to empty list (will store probability tensors)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize with a small tensor to establish device
    cls_memory = {
        i: [torch.ones(1, num_classes, device=device)] for i in range(num_classes)
    }
    return cls_memory


def downsample_for_memory(
    prob: torch.Tensor, label: torch.Tensor, target_size: int = 64
) -> tuple:
    """
    Downsample probability maps and labels for efficient memory storage.

    Args:
        prob: Probability map of shape (B, C, H, W)
        label: Label map of shape (B, H, W)
        target_size: Target spatial size for downsampling

    Returns:
        Tuple of (flattened_prob, flattened_label) ready for memory update
    """
    b, c, h, w = prob.shape

    # Downsample probability: (B, C, H, W) -> (B, C, target_size, target_size) -> (N, C)
    prob_bar = F.interpolate(prob, size=(target_size, target_size), mode="nearest")
    prob_bar = prob_bar.permute(0, 2, 3, 1).reshape(-1, c)

    # Downsample label: (B, H, W) -> (B, target_size, target_size) -> (N,)
    label_bar = F.interpolate(
        label.float().unsqueeze(1), size=(target_size, target_size), mode="nearest"
    )
    label_bar = label_bar.squeeze(1).reshape(-1).long()

    return prob_bar, label_bar
