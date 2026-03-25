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
    del pseudo_conf, conf_thresh, ignore_index
    labels = torch.unique(pseudo_labels)
    if labels.numel() == 0:
        return torch.zeros_like(pseudo_labels, dtype=torch.float32)
    selected = labels[torch.randperm(labels.numel(), device=pseudo_labels.device)[: labels.numel() // 2]]
    return (pseudo_labels.unsqueeze(-1) == selected).any(dim=-1).float()


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
    del ignore_mask, ignore_index, conf_thresh
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

    for i in range(batch):
        mix_mask = generate_class_mask(pseudo_label[i])
        mix_masks.append(mix_mask.unsqueeze(0))
        j = (i + 1) % batch
        mix = mix_mask.unsqueeze(0)
        out_img_w.append((img_w[i] * mix + img_w[j] * (1 - mix)).unsqueeze(0))
        out_img_s.append((img_s[i] * mix + img_s[j] * (1 - mix)).unsqueeze(0))
        out_label.append((pseudo_label[i] * mix_mask + pseudo_label[j] * (1 - mix_mask)).unsqueeze(0))
        out_logit.append((pseudo_logit[i] * mix_mask + pseudo_logit[j] * (1 - mix_mask)).unsqueeze(0))
        out_entropy.append((entropy[i] * mix_mask + entropy[j] * (1 - mix_mask)).unsqueeze(0))

    return {
        "img_w": torch.cat(out_img_w, dim=0),
        "img_s": torch.cat(out_img_s, dim=0),
        "pseudo_label": torch.cat(out_label, dim=0).long(),
        "pseudo_logit": torch.cat(out_logit, dim=0),
        "entropy": torch.cat(out_entropy, dim=0),
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


def _sample_negative_features(
    sample_class_counts: torch.Tensor,
    bank: SegMindMemoryBank,
    feat_dim: int,
    num_queries: int,
    num_negative: int,
    device: torch.device,
) -> torch.Tensor:
    negatives = torch.zeros((num_queries, num_negative, feat_dim), device=device)
    for query_idx in range(sample_class_counts.shape[0]):
        chunks = []
        for class_idx, sample_count in enumerate(sample_class_counts[query_idx].tolist()):
            if sample_count <= 0 or bank.queues[class_idx].shape[0] == 0:
                continue
            sample_idx = torch.randint(
                low=0,
                high=bank.queues[class_idx].shape[0],
                size=(sample_count,),
                device=device,
            )
            chunks.append(bank.queues[class_idx][sample_idx].to(device))
        if not chunks:
            continue
        chunk = torch.cat(chunks, dim=0)
        negatives[query_idx, : chunk.shape[0]] = chunk
    return negatives


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
    num_classes = prob.shape[1]
    feat_dim = feat.shape[1]
    feat_hw = feat.permute(0, 2, 3, 1)
    labels_small = F.interpolate(labels.float().unsqueeze(1), size=feat.shape[-2:], mode="nearest").squeeze(1).long()
    prob_small = F.interpolate(prob, size=feat.shape[-2:], mode="bilinear", align_corners=True)
    device = feat.device

    valid_classes = []
    batch_means = []
    hard_features = []
    queue_means = torch.zeros((num_classes, feat_dim), device=device)

    for class_idx in range(num_classes):
        valid_mask = labels_small == class_idx
        if valid_mask.sum() == 0:
            continue
        valid_classes.append(class_idx)
        class_feat = feat_hw[valid_mask]
        dequeue_and_enqueue(class_feat, bank, class_idx)
        batch_means.append(torch.mean(class_feat, dim=0, keepdim=True))
        hard_features.append(feat_hw[(prob_small[:, class_idx] < query_threshold) & valid_mask])
        if bank.queues[class_idx].shape[0] > 0:
            queue_means[class_idx] = torch.mean(bank.queues[class_idx], dim=0).to(device)

    if not valid_classes:
        return feat.sum() * 0.0

    total_loss = feat.sum() * 0.0
    for valid_idx, class_idx in enumerate(valid_classes):
        class_hard_features = hard_features[valid_idx]
        if class_hard_features.shape[0] == 0:
            continue
        query_idx = torch.randint(class_hard_features.shape[0], (num_queries,), device=device)
        query_features = class_hard_features[query_idx]

        mean_similarity = F.cosine_similarity(batch_means[valid_idx].to(device), queue_means, dim=1)
        mean_similarity = torch.cat((mean_similarity[:class_idx], mean_similarity[class_idx + 1 :]))
        if mean_similarity.numel() == 0:
            continue
        negative_probs = torch.softmax(mean_similarity, dim=0)
        full_probs = torch.zeros(num_classes, device=device)
        if class_idx > 0:
            full_probs[:class_idx] = negative_probs[:class_idx]
        if class_idx + 1 < num_classes:
            full_probs[class_idx + 1 :] = negative_probs[class_idx:]
        if full_probs.sum() <= 0:
            continue
        sampler = torch.distributions.Categorical(probs=full_probs)
        sampled_classes = sampler.sample((num_queries, num_negative))
        sampled_class_counts = torch.stack(
            [(sampled_classes == cls_idx).sum(dim=1) for cls_idx in range(num_classes)],
            dim=1,
        )
        negative_features = _sample_negative_features(
            sampled_class_counts,
            bank,
            feat_dim=feat_dim,
            num_queries=num_queries,
            num_negative=num_negative,
            device=device,
        )
        positive_features = batch_means[valid_idx].to(device).unsqueeze(0).repeat(num_queries, 1, 1)
        all_features = torch.cat((positive_features, negative_features), dim=1)
        logits = F.cosine_similarity(query_features.unsqueeze(1), all_features, dim=2)
        total_loss = total_loss + F.cross_entropy(
            logits / temperature,
            torch.zeros(num_queries, dtype=torch.long, device=device),
        )

    return total_loss / len(valid_classes)


@torch.no_grad()
def gather_pseudo_from_teacher(model_ema, img_l_w: torch.Tensor, img_u_w: torch.Tensor):
    outputs = model_ema(torch.cat((img_l_w, img_u_w), dim=0), return_aux=True)
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
