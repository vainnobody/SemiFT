from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SegMindMemoryBank:
    queues: List[torch.Tensor]
    ptrs: List[int]
    bank_size: int

    def state_dict(self) -> Dict[str, object]:
        return {
            "queues": [queue.detach().cpu() for queue in self.queues],
            "ptrs": list(self.ptrs),
            "bank_size": self.bank_size,
        }

    def load_state_dict(self, state_dict: Dict[str, object], device: torch.device):
        queues = state_dict.get("queues", [])
        self.bank_size = int(state_dict.get("bank_size", self.bank_size))
        self.ptrs = [int(v) for v in state_dict.get("ptrs", self.ptrs)]
        for idx, queue in enumerate(queues):
            self.queues[idx] = queue.to(device)


def create_memory_bank(num_classes: int, proj_dim: int, bank_size: int, device: torch.device) -> SegMindMemoryBank:
    queues = [torch.zeros((0, proj_dim), device=device) for _ in range(num_classes)]
    ptrs = [0 for _ in range(num_classes)]
    return SegMindMemoryBank(queues=queues, ptrs=ptrs, bank_size=bank_size)


def entropy_map_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    return torch.sum(-probs * torch.log(probs.clamp_min(1e-8)), dim=1)


def generate_grid_mask(batch: int, height: int, width: int, mask_gap: int, mask_rate: float, device: torch.device) -> torch.Tensor:
    if height % mask_gap != 0 or width % mask_gap != 0:
        raise ValueError("height and width must be divisible by mask_gap")
    gh, gw = height // mask_gap, width // mask_gap
    masks = []
    for _ in range(batch):
        perm = torch.randperm(gh * gw, device=device).reshape(gh, gw)
        threshold = int(gh * gw * mask_rate)
        small = (perm >= threshold).float().unsqueeze(0).unsqueeze(0)
        masks.append(F.interpolate(small, size=(height, width), mode="nearest"))
    return torch.cat(masks, dim=0)


def generate_class_mask(
    pseudo_labels: torch.Tensor,
    pseudo_conf: torch.Tensor | None = None,
    conf_thresh: float = 0.0,
    ignore_index: int = 255,
) -> torch.Tensor:
    valid = pseudo_labels != ignore_index
    if pseudo_conf is not None:
        valid = valid & (pseudo_conf >= conf_thresh)
    labels = torch.unique(pseudo_labels[valid])
    if labels.numel() == 0:
        return torch.zeros_like(pseudo_labels, dtype=torch.float32)
    selected = labels[torch.randperm(labels.numel(), device=pseudo_labels.device)[: max(1, labels.numel() // 2)]]
    mask = (pseudo_labels.unsqueeze(-1) == selected).any(dim=-1) & valid
    return mask.float()


def class_mix_batch(
    img_w: torch.Tensor | None = None,
    img_s: torch.Tensor | None = None,
    pseudo_label: torch.Tensor | None = None,
    pseudo_logit: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    ignore_mask: torch.Tensor | None = None,
    ignore_index: int = 255,
    conf_thresh: float = 0.0,
    img_u_w: torch.Tensor | None = None,
):
    if img_w is None:
        img_w = img_u_w
    if img_w is None:
        raise TypeError("class_mix_batch() missing required argument: 'img_w'")
    if img_s is None or pseudo_label is None or pseudo_logit is None or entropy is None:
        raise TypeError("class_mix_batch() requires img_s, pseudo_label, pseudo_logit, and entropy")
    batch = img_w.shape[0]
    mix_masks = []
    out_img_w, out_img_s = [], []
    out_label, out_logit, out_entropy = [], [], []
    out_ignore = [] if ignore_mask is not None else None

    for i in range(batch):
        mix_mask = generate_class_mask(
            pseudo_label[i],
            pseudo_conf=None if pseudo_logit is None else pseudo_logit[i],
            conf_thresh=conf_thresh,
            ignore_index=ignore_index,
        )
        mix_masks.append(mix_mask.unsqueeze(0))
        j = (i + 1) % batch
        mix = mix_mask.unsqueeze(0)
        out_img_w.append((img_w[i] * mix + img_w[j] * (1 - mix)).unsqueeze(0))
        out_img_s.append((img_s[i] * mix + img_s[j] * (1 - mix)).unsqueeze(0))
        out_label.append((pseudo_label[i] * mix_mask + pseudo_label[j] * (1 - mix_mask)).unsqueeze(0))
        out_logit.append((pseudo_logit[i] * mix_mask + pseudo_logit[j] * (1 - mix_mask)).unsqueeze(0))
        out_entropy.append((entropy[i] * mix_mask + entropy[j] * (1 - mix_mask)).unsqueeze(0))
        if ignore_mask is not None:
            mixed_ignore = torch.where(mix_mask.bool(), ignore_mask[i], ignore_mask[j])
            out_ignore.append(mixed_ignore.unsqueeze(0))

    return {
        "img_w": torch.cat(out_img_w, dim=0),
        "img_s": torch.cat(out_img_s, dim=0),
        "pseudo_label": torch.cat(out_label, dim=0).long(),
        "pseudo_logit": torch.cat(out_logit, dim=0),
        "entropy": torch.cat(out_entropy, dim=0),
        "ignore_mask": None if out_ignore is None else torch.cat(out_ignore, dim=0),
        "mix_mask": torch.cat(mix_masks, dim=0),
    }


@torch.no_grad()
def dequeue_and_enqueue(keys: torch.Tensor, bank: SegMindMemoryBank, class_idx: int):
    if keys.numel() == 0:
        return
    queue = torch.cat((bank.queues[class_idx], keys.detach()), dim=0)
    if queue.shape[0] > bank.bank_size:
        queue = queue[-bank.bank_size :]
    bank.queues[class_idx] = queue
    bank.ptrs[class_idx] = min(queue.shape[0], bank.bank_size)


def _sample_negatives(bank: SegMindMemoryBank, num_classes: int, exclude_class: int, num_negative: int, device: torch.device):
    candidates = []
    for cls_idx in range(num_classes):
        if cls_idx == exclude_class:
            continue
        if bank.queues[cls_idx].shape[0] > 0:
            candidates.append(bank.queues[cls_idx])
    if not candidates:
        return None
    negatives = torch.cat(candidates, dim=0)
    if negatives.shape[0] <= num_negative:
        return negatives.to(device)
    idx = torch.randint(0, negatives.shape[0], (num_negative,), device=device)
    return negatives[idx].to(device)


def segmind_contrastive_loss(
    feat: torch.Tensor,
    labels: torch.Tensor,
    prob: torch.Tensor,
    bank: SegMindMemoryBank,
    query_threshold: float,
    temperature: float,
    num_queries: int,
    num_negative: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    feat = F.normalize(feat, dim=1)
    num_classes = prob.shape[1]
    feat_hw = feat.permute(0, 2, 3, 1)
    labels_small = F.interpolate(labels.float().unsqueeze(1), size=feat.shape[-2:], mode="nearest").squeeze(1).long()
    prob_small = F.interpolate(prob, size=feat.shape[-2:], mode="bilinear", align_corners=True)
    device = feat.device
    losses = []

    for cls_idx in range(num_classes):
        valid = labels_small == cls_idx
        if valid.sum() == 0:
            continue
        class_feat = feat_hw[valid]
        dequeue_and_enqueue(class_feat, bank, cls_idx)

        hard_mask = valid & (prob_small[:, cls_idx] < query_threshold)
        hard_feat = feat_hw[hard_mask]
        if hard_feat.shape[0] == 0 or bank.queues[cls_idx].shape[0] == 0:
            continue

        q_idx = torch.randint(0, hard_feat.shape[0], (min(num_queries, hard_feat.shape[0]),), device=device)
        queries = hard_feat[q_idx]
        positive = F.normalize(bank.queues[cls_idx].mean(dim=0, keepdim=True).to(device), dim=1)
        negatives = _sample_negatives(bank, num_classes, cls_idx, num_negative, device)
        if negatives is None or negatives.shape[0] == 0:
            continue
        negatives = F.normalize(negatives, dim=1)

        pos_logits = torch.matmul(queries, positive.t())
        neg_logits = torch.matmul(queries, negatives.t())
        logits = torch.cat((pos_logits, neg_logits), dim=1) / temperature
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits, target))

    if not losses:
        return feat.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def gather_pseudo_from_teacher(model_ema, img_l_w: torch.Tensor, img_u_w: torch.Tensor):
    was_training = model_ema.training
    model_ema.eval()
    try:
        outputs = model_ema(torch.cat((img_l_w, img_u_w), dim=0), return_aux=True)
    finally:
        if was_training:
            model_ema.train()
    logits = outputs["seg_logits"]
    probs = logits.softmax(dim=1)
    pseudo_logit, pseudo_label = probs[img_l_w.shape[0] :].max(dim=1)
    entropy = entropy_map_from_logits(logits)[img_l_w.shape[0] :]
    return logits, pseudo_logit, pseudo_label, entropy


def percentile_entropy_mask(entropy: torch.Tensor, valid_mask: torch.Tensor, percent: float) -> torch.Tensor:
    valid_entropy = entropy[valid_mask]
    if valid_entropy.numel() == 0:
        return torch.zeros_like(valid_mask, dtype=torch.bool)
    threshold = np.percentile(valid_entropy.detach().cpu().numpy(), percent)
    return (entropy <= threshold) & valid_mask
