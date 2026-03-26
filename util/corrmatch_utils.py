import torch
import torch.distributed as dist


def _gather_cat(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


class ThreshController:
    def __init__(self, nclass, momentum=0.999, thresh_init=0.85):
        self.nclass = nclass
        self.momentum = momentum
        self.thresh_global = torch.tensor(float(thresh_init), device="cuda")

    @torch.no_grad()
    def _compute_new_global(self, pred, ignore_mask=None):
        pred = _gather_cat(pred)
        if ignore_mask is not None:
            ignore_mask = _gather_cat(ignore_mask)

        mask_pred = pred.argmax(dim=1)
        pred_conf = pred.softmax(dim=1).max(dim=1)[0]
        unique_cls = torch.unique(mask_pred)

        cls_values = []
        for cls in unique_cls:
            cls_map = mask_pred == cls
            if ignore_mask is not None:
                cls_map = cls_map & (ignore_mask != 255)
            if cls_map.any():
                cls_values.append(pred_conf[cls_map].max())

        if not cls_values:
            return None
        return torch.stack(cls_values).mean()

    @torch.no_grad()
    def thresh_update(self, pred, ignore_mask=None, update_g=True):
        new_global = self._compute_new_global(pred, ignore_mask)
        if update_g and new_global is not None:
            self.thresh_global = (
                self.momentum * self.thresh_global + (1 - self.momentum) * new_global
            )
        return self.thresh_global

    def get_thresh_global(self):
        return self.thresh_global


@torch.no_grad()
def apply_region_propagation(
    mask,
    corr_map,
    conf_filter,
    thresh_global,
):
    propagated_mask = mask.clone()
    expanded_filter = conf_filter.clone()
    batch_size, num_regions = corr_map.shape[:2]

    for batch_idx in range(batch_size):
        for region_idx in range(num_regions):
            region_all = corr_map[batch_idx, region_idx].bool()
            if region_all.sum() == 0:
                continue

            region_conf = region_all & expanded_filter[batch_idx]
            high_conf_ratio = region_conf.sum().float() / region_all.sum().float().clamp_min(1.0)
            if high_conf_ratio < thresh_global:
                continue

            labels, counts = torch.unique(
                propagated_mask[batch_idx][region_conf], return_counts=True
            )
            if labels.numel() == 0:
                continue

            dominant_ratio = counts.max().float() / counts.sum().float().clamp_min(1.0)
            if dominant_ratio > thresh_global:
                dominant_label = labels[counts.argmax()]
                propagated_mask[batch_idx][region_all] = dominant_label
                expanded_filter[batch_idx] = expanded_filter[batch_idx] | region_all

    return propagated_mask, expanded_filter
