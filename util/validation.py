import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from util.utils import AverageMeter, intersectionAndUnion


def extract_validation_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        if "out" in output:
            return output["out"]
        raise TypeError(f"Unsupported validation output dict keys: {list(output.keys())}")
    if isinstance(output, (list, tuple)):
        if not output:
            raise TypeError("Validation output list/tuple is empty.")
        return output[0]
    raise TypeError(f"Unsupported validation output type: {type(output)!r}")


def sync_histograms(intersection, union, target_area, device):
    if not dist.is_available() or not dist.is_initialized():
        return intersection, union, target_area

    reduced = []
    for value in (intersection, union, target_area):
        tensor = torch.as_tensor(value, device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced.append(tensor.cpu().numpy())
    return tuple(reduced)


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()

    for x, y, _ in valid_loader:
        x = x.cuda()
        if cfg["eval_mode"] == "slide_window":
            batch_size, _, h, w = x.shape
            final = torch.zeros(batch_size, cfg["nclass"], h, w, device=x.device)
            count = torch.zeros(batch_size, 1, h, w, device=x.device)

            crop_size = cfg["crop_size"]
            if isinstance(crop_size, int):
                crop_h = crop_w = crop_size
            else:
                crop_h, crop_w = crop_size

            step = cfg.get("stride", 510)
            if isinstance(step, int):
                step_h = step_w = step
            else:
                step_h, step_w = step

            h_starts = list(range(0, max(h - crop_h, 0) + 1, step_h))
            w_starts = list(range(0, max(w - crop_w, 0) + 1, step_w))
            if not h_starts:
                h_starts = [0]
            if not w_starts:
                w_starts = [0]
            last_h = max(h - crop_h, 0)
            last_w = max(w - crop_w, 0)
            if h_starts[-1] != last_h:
                h_starts.append(last_h)
            if w_starts[-1] != last_w:
                w_starts.append(last_w)

            for hs in h_starts:
                for ws in w_starts:
                    he = min(hs + crop_h, h)
                    we = min(ws + crop_w, w)
                    sub_input = x[:, :, hs:he, ws:we]
                    mask = extract_validation_logits(model(sub_input))
                    final[:, :, hs:he, ws:we] += mask
                    count[:, :, hs:he, ws:we] += 1

            final = final / count.clamp_min(1.0)
            o = final.argmax(dim=1)

        elif cfg["eval_mode"] == "resize":
            original_shape = x.shape[-2:]
            resized_x = F.interpolate(
                x, size=cfg["crop_size"], mode="bilinear", align_corners=True
            )
            resized_o = extract_validation_logits(model(resized_x))
            o = F.interpolate(
                resized_o, size=original_shape, mode="bilinear", align_corners=True
            )
            o = o.argmax(dim=1)

        else:
            o = extract_validation_logits(model(x))
            o = o.max(1)[1]

        gray = np.uint8(o.cpu().numpy())
        target = np.array(y, dtype=np.int32)
        intersection, union, target_area = intersectionAndUnion(
            gray, target, cfg["nclass"], cfg["ignore_index"]
        )
        intersection, union, target_area = sync_histograms(
            intersection, union, target_area, x.device
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target_area)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    if cfg["dataset"] == "iSAID":
        mIoU = np.mean(iou_class[1:]) * 100.0
    else:
        mIoU = np.nanmean(iou_class) * 100.0

    return mIoU, iou_class
