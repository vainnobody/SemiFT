import torch
import torch.nn.functional as F


def apply_structured_feature_perturbation(feature_map, feature_perturb):
    """
    Apply geometry-/confidence-aware structured perturbation on feature maps.

    Args:
        feature_map: [B, C, H, W]
        feature_perturb: dict with at least `score_map`

    Returns:
        Perturbed feature map with the same shape as input.
    """
    if not feature_perturb:
        return feature_map

    score_map = feature_perturb.get("score_map")
    if score_map is None:
        return feature_map

    if score_map.dim() == 3:
        score_map = score_map.unsqueeze(1)

    score_map = score_map.to(device=feature_map.device, dtype=feature_map.dtype)
    score_map = F.interpolate(
        score_map,
        size=feature_map.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp_(0.0, 1.0)

    if feature_perturb.get("smooth_score", True):
        score_map = F.avg_pool2d(score_map, kernel_size=3, stride=1, padding=1)

    mask_strength = float(feature_perturb.get("mask_strength", 0.6))
    shift_strength = float(feature_perturb.get("shift_strength", 0.15))
    stable_floor = float(feature_perturb.get("stable_floor", 0.35))
    kernel_size = int(feature_perturb.get("local_kernel", 3))
    padding = kernel_size // 2

    local_context = F.avg_pool2d(
        feature_map, kernel_size=kernel_size, stride=1, padding=padding
    )
    structural_residual = feature_map - local_context
    residual_scale = structural_residual.abs().mean(dim=(2, 3), keepdim=True) + 1e-6
    structural_residual = structural_residual / residual_scale

    retain_gate = (1.0 - mask_strength * score_map).clamp(min=stable_floor, max=1.0)
    perturbed = feature_map * retain_gate
    perturbed = perturbed + shift_strength * score_map * structural_residual

    return perturbed
