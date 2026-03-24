from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SegMindQueueState:
    banks: list
    ptrs: list
    bank_size: int


def generate_block_mask(height, width, mask_gap=16, mask_rate=0.75, device=None):
    if height % mask_gap != 0 or width % mask_gap != 0:
        raise ValueError("height and width must be divisible by mask_gap")
    h_steps, w_steps = height // mask_gap, width // mask_gap
    mask_small = torch.randperm(h_steps * w_steps, device=device, dtype=torch.float32)
    mask_small = mask_small.reshape(h_steps, w_steps)
    threshold = h_steps * w_steps * mask_rate
    mask_small = (mask_small >= threshold).float()
    return F.interpolate(
        mask_small.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="nearest",
    ).squeeze(0)


def get_batch_mask_tensor(shape, mask_gap=16, mask_rate=0.75, device=None):
    batch_size, _, height, width = shape
    masks = [
        generate_block_mask(height, width, mask_gap=mask_gap, mask_rate=mask_rate, device=device)
        for _ in range(batch_size)
    ]
    return torch.stack(masks, dim=0)


def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)
    if labels.numel() <= 1:
        return torch.ones_like(pseudo_labels, dtype=torch.float32)
    selected = labels[torch.randperm(labels.numel(), device=labels.device)[: labels.numel() // 2]]
    return (pseudo_labels.unsqueeze(-1) == selected).any(dim=-1).float()


def classmix_batch(*tensors, labels):
    """Mix each sample with its next neighbor using a class mask from labels."""
    mixed_tensors = [[] for _ in tensors]
    mixed_masks = []
    batch_size = labels.shape[0]

    for idx in range(batch_size):
        mix_mask = generate_class_mask(labels[idx])
        mixed_masks.append(mix_mask)
        partner = (idx + 1) % batch_size
        for tensor_list, tensor in zip(mixed_tensors, tensors):
            if tensor.dim() == 4:
                cur_mask = mix_mask.unsqueeze(0)
            else:
                cur_mask = mix_mask
            mixed = tensor[idx] * cur_mask + tensor[partner] * (1.0 - cur_mask)
            tensor_list.append(mixed.unsqueeze(0))

    outputs = [torch.cat(items, dim=0) for items in mixed_tensors]
    return (*outputs, torch.stack(mixed_masks, dim=0))


def resize_labels_to_shape(labels, target_shape):
    return F.interpolate(
        labels.float().unsqueeze(1),
        size=target_shape,
        mode="nearest",
    ).squeeze(1).long()


def init_queue_state(num_classes, feat_dim, bank_size):
    return SegMindQueueState(
        banks=[torch.zeros((0, feat_dim), dtype=torch.float32) for _ in range(num_classes)],
        ptrs=[torch.zeros(1, dtype=torch.long) for _ in range(num_classes)],
        bank_size=bank_size,
    )


def _enqueue_class_features(queue_state, class_idx, class_feats):
    if class_feats.numel() == 0:
        return
    bank = queue_state.banks[class_idx]
    bank = torch.cat((bank, class_feats.detach().cpu().float()), dim=0)
    if bank.shape[0] >= queue_state.bank_size:
        bank = bank[-queue_state.bank_size :]
        ptr = queue_state.bank_size
    else:
        ptr = min(int(queue_state.ptrs[class_idx].item()) + class_feats.shape[0], queue_state.bank_size)
    queue_state.banks[class_idx] = bank
    queue_state.ptrs[class_idx][0] = ptr


def serialize_queue_state(queue_state):
    return {
        "banks": [bank.clone() for bank in queue_state.banks],
        "ptrs": [ptr.clone() for ptr in queue_state.ptrs],
        "bank_size": queue_state.bank_size,
    }


def load_queue_state(payload):
    return SegMindQueueState(
        banks=[bank.clone().float() for bank in payload["banks"]],
        ptrs=[ptr.clone().long() for ptr in payload["ptrs"]],
        bank_size=int(payload["bank_size"]),
    )


def compute_reconstruction_loss(recon, target, mask):
    missing = ~mask.bool()
    if missing.sum() == 0:
        return recon.new_zeros(())
    recon_flat = recon.permute(0, 2, 3, 1)[missing]
    target_flat = target.permute(0, 2, 3, 1)[missing]
    return F.mse_loss(recon_flat, target_flat)


def compute_masked_segmentation_loss(logits, labels, mask, ignore_index):
    masked_labels = labels.clone()
    masked_labels[mask.bool()] = ignore_index
    return F.cross_entropy(logits, masked_labels, ignore_index=ignore_index)


def compute_contrastive_loss(
    proj_feat,
    labels,
    probs,
    queue_state,
    query_threshold,
    temperature,
    num_query,
    num_negative,
    ignore_index,
):
    # Keep gradients on current-batch query features so loss_c updates the
    # projection branch, matching SegMind's contrastive objective.
    feat = F.normalize(proj_feat.float(), dim=1)
    feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
    labels = resize_labels_to_shape(labels.detach(), proj_feat.shape[-2:]).reshape(-1)
    probs = F.interpolate(
        probs.detach().float(),
        size=proj_feat.shape[-2:],
        mode="bilinear",
        align_corners=True,
    )
    probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])

    valid_mask = labels != ignore_index
    feat = feat[valid_mask]
    labels = labels[valid_mask]
    probs = probs[valid_mask]
    if feat.numel() == 0:
        return proj_feat.new_zeros(())

    feat_dim = feat.shape[1]
    device = proj_feat.device
    losses = []
    class_means = {}

    for class_idx in range(len(queue_state.banks)):
        class_mask = labels == class_idx
        if not class_mask.any():
            continue
        class_feats = feat[class_mask]
        class_feats_detached = class_feats.detach()
        _enqueue_class_features(queue_state, class_idx, class_feats_detached)
        class_means[class_idx] = class_feats_detached.mean(dim=0, keepdim=True)

    for class_idx, class_mean in class_means.items():
        class_mask = labels == class_idx
        hard_mask = class_mask & (probs[:, class_idx] < query_threshold)
        hard_feats = feat[hard_mask]
        if hard_feats.numel() == 0:
            continue

        query_idx = torch.randint(
            hard_feats.shape[0],
            size=(num_query,),
            device=hard_feats.device,
        )
        query_feat = hard_feats[query_idx]

        negative_classes = []
        negative_scores = []
        for other_idx, other_mean in class_means.items():
            if other_idx == class_idx or queue_state.banks[other_idx].shape[0] == 0:
                continue
            negative_classes.append(other_idx)
            negative_scores.append(
                F.cosine_similarity(class_mean.to(device), other_mean.to(device), dim=1)
            )
        if not negative_classes:
            continue

        negative_scores = torch.stack(negative_scores).squeeze(1)
        negative_probs = torch.softmax(negative_scores, dim=0)
        sampled_class_indices = torch.multinomial(
            negative_probs,
            num_samples=num_query * num_negative,
            replacement=True,
        ).view(num_query, num_negative)

        negative_feat = torch.zeros(
            (num_query, num_negative, feat_dim),
            device=device,
            dtype=query_feat.dtype,
        )
        for row in range(num_query):
            for col in range(num_negative):
                neg_class = negative_classes[int(sampled_class_indices[row, col].item())]
                bank = queue_state.banks[neg_class]
                neg_idx = torch.randint(bank.shape[0], size=(1,)).item()
                negative_feat[row, col] = bank[neg_idx].to(device)

        positive_feat = class_mean.to(device).expand(num_query, 1, feat_dim)
        all_feat = torch.cat((positive_feat, negative_feat), dim=1)
        logits = F.cosine_similarity(query_feat.unsqueeze(1), all_feat, dim=2)
        targets = torch.zeros(num_query, dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits / temperature, targets))

    if not losses:
        return proj_feat.new_zeros(())
    return torch.stack(losses).mean()
