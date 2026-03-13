"""
FixMatch-RGCR v2: engineering-stable variant of fixmatch_rgcr.py.

Changes vs fixmatch_rgcr.py:
- safer config defaults for ignore_index / eval_mode / workers
- distributed validation with all_reduce
- no per-iteration barrier
- rank0-only visualization / logging / writer / checkpoints
- separate best student and best EMA checkpoints
"""

import argparse
from copy import deepcopy
from datetime import datetime
import logging
import os
import pprint

import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.semi_rvs import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt_rankmatch import DPT_RankMatch
from model.semseg.rgcr_utils import scale_back
from util.classes import CLASSES
from util.dist_helper import setup_distributed
from util.focal import FocalLoss
from util.ohem import ProbOhemCrossEntropy2d
from util.train_utils import confidence_weighted_loss
from util.utils import AverageMeter, count_params, init_log, intersectionAndUnion
from util.viz import Visualizer


def get_parser():
    parser = argparse.ArgumentParser(
        description="FixMatch-RGCR v2 for Semi-Supervised Semantic Segmentation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


@torch.no_grad()
def validation_distributed(cfg, model, valid_loader, eval_mode, ignore_index):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()

    for x, y, _ in valid_loader:
        x = x.cuda(non_blocking=True)

        if eval_mode == "slide_window":
            bsz, _, h, w = x.shape
            final = torch.zeros(bsz, cfg["nclass"], h, w, device=x.device)
            size = cfg["crop_size"]
            step = 510 if cfg["dataset"] == "cityscapes" else max(size * 2 // 3, 1)
            row = 0
            while row <= int(h / step):
                col = 0
                while col <= int(w / step):
                    h0 = min(row * step, h - size)
                    w0 = min(col * step, w - size)
                    sub_input = x[:, :, h0 : min(h0 + size, h), w0 : min(w0 + size, w)]
                    pred = model(sub_input)[0]
                    final[:, :, h0 : min(h0 + size, h), w0 : min(w0 + size, w)] += pred
                    col += 1
                row += 1
            pred_label = final.argmax(dim=1)
        elif eval_mode == "resize":
            original_shape = x.shape[-2:]
            resized_x = F.interpolate(
                x, size=cfg["crop_size"], mode="bilinear", align_corners=True
            )
            resized_o = model(resized_x)[0]
            pred_label = F.interpolate(
                resized_o, size=original_shape, mode="bilinear", align_corners=True
            ).argmax(dim=1)
        else:
            pred_label = model(x)[0].argmax(dim=1)

        gray = np.uint8(pred_label.cpu().numpy())
        target = np.array(y, dtype=np.int32)
        intersection, union, target_area = intersectionAndUnion(
            gray, target, cfg["nclass"], ignore_index
        )

        reduced_intersection = torch.from_numpy(intersection).to(x.device)
        reduced_union = torch.from_numpy(union).to(x.device)
        reduced_target = torch.from_numpy(target_area).to(x.device)

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    if cfg["dataset"] == "iSAID":
        miou = np.mean(iou_class[1:]) * 100.0
    else:
        miou = np.nanmean(iou_class) * 100.0

    return miou, iou_class


def main(args, cfg):
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    is_main = rank == 0

    ignore_index = cfg.get("ignore_index", 255)
    eval_mode = cfg.get(
        "eval_mode",
        "slide_window" if cfg["dataset"] == "cityscapes" else "original",
    )
    num_workers = cfg.get("workers", 4)

    writer = None
    viz = None
    if is_main:
        os.makedirs(args.save_path, exist_ok=True)
        writer = SummaryWriter(args.save_path)
        all_args = {
            **cfg,
            **vars(args),
            "ngpus": world_size,
            "resolved_ignore_index": ignore_index,
            "resolved_eval_mode": eval_mode,
            "resolved_workers": num_workers,
        }
        logger.info("{}\n".format(pprint.pformat(all_args)))

        filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz = Visualizer(save_dir=f"./viz/{filename}", dataset=cfg["dataset"])

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        "small": {
            "encoder_size": "small",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "base": {
            "encoder_size": "base",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "large": {
            "encoder_size": "large",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "giant": {
            "encoder_size": "giant",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    backbone_size = cfg["backbone"].split("_")[-1]
    backbone_version = cfg["backbone"].split("_")[0]

    model = DPT_RankMatch(
        **{**model_configs[backbone_size], "nclass": cfg["nclass"]},
        backbone_version=backbone_version,
    )

    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)

    if cfg.get("lock_backbone", False):
        model.lock_backbone()

    optimizer = AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": cfg["lr"],
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "backbone" not in name
                ],
                "lr": cfg["lr"] * cfg["lr_multi"],
            },
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    if is_main:
        logger.info("Total params: {:.1f}M".format(count_params(model)))
        logger.info("Encoder params: {:.1f}M".format(count_params(model.backbone)))
        logger.info("Decoder params: {:.1f}M\n".format(count_params(model.head)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )

    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(
            local_rank
        )
    elif cfg["criterion"]["name"] == "FocalLoss":
        criterion_l = FocalLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    else:
        raise NotImplementedError(
            f'{cfg["criterion"]["name"]} criterion is not implemented'
        )

    criterion_u = nn.CrossEntropyLoss(reduction="none", ignore_index=255).cuda(
        local_rank
    )

    trainset_u = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        size=cfg["crop_size"],
        ignore_value=ignore_index,
        id_path=args.unlabeled_id_path,
    )
    trainset_l = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        size=cfg["crop_size"],
        ignore_value=ignore_index,
        id_path=args.labeled_id_path,
        nsample=len(trainset_u.ids),
    )
    valset = ValDataset(
        cfg["dataset"], cfg["data_root"], "val", ignore_value=ignore_index
    )

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        sampler=trainsampler_l,
    )

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        sampler=trainsampler_u,
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=max(1, min(num_workers, 2)),
        drop_last=False,
        sampler=valsampler,
    )

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    conf_thresh = cfg["conf_thresh"]

    latest_path = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path)
        model.load_state_dict(checkpoint["model"])
        model_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        previous_best_ema = checkpoint.get("previous_best_ema", 0.0)
        best_epoch = checkpoint.get("best_epoch", 0)
        best_epoch_ema = checkpoint.get("best_epoch_ema", 0)
        if is_main:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    for epoch in range(epoch + 1, cfg["epochs"]):
        if is_main:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, "
                "EMA: {:.2f} @epoch-{:}".format(
                    epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema
                )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_u_s = AverageMeter()
        total_loss_u_rvs = AverageMeter()
        total_loss_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        valloader.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        model.train()

        for i, (
            (img_x, mask_x, img_x_c, mask_x_c),
            (img_u_w, img_u_s, img_u_c, ignore_mask, cutmix_box, _, box, mask_c),
        ) in enumerate(loader):
            del img_x_c, mask_x_c

            img_x, mask_x = img_x.cuda(non_blocking=True), mask_x.cuda(non_blocking=True)
            img_u_w = img_u_w.cuda(non_blocking=True)
            img_u_s = img_u_s.cuda(non_blocking=True)
            img_u_c = img_u_c.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)
            cutmix_box = cutmix_box.cuda(non_blocking=True)
            mask_c = mask_c.cuda(non_blocking=True)
            box = box.cuda(non_blocking=True)

            iters = epoch * len(trainloader_u) + i

            with torch.no_grad():
                pred_u_w_ema, feat_u_w_ema = model_ema(img_u_w)
                pred_u_w_ema = pred_u_w_ema.detach()
                feat_u_w_ema = feat_u_w_ema.detach()
                conf_u_w = pred_u_w_ema.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w_ema.argmax(dim=1)

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            preds, preds_fp, feats = model(torch.cat((img_x, img_u_w)), need_fp=True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            _, feat_u_w = feats.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            del pred_u_w, feat_u_w, feat_u_w_ema

            cutmix_mask = cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1
            img_u_s[cutmix_mask] = img_u_s.flip(0)[cutmix_mask]
            pred_u_s = model(img_u_s)[0]

            pred_u_rvs, feat_u_rvs = model(img_u_c)
            pred_recovered, valid_masks_pred = scale_back(
                pred_u_rvs, mask_c, cfg["crop_size"], box
            )
            del feat_u_rvs

            valid_masks_pred_sq = valid_masks_pred.squeeze(1)
            mask_u_w_rvs = mask_u_w.clone()
            mask_u_w_rvs[valid_masks_pred_sq == 0] = 255

            ignore_mask_rvs = ignore_mask.clone()
            ignore_mask_rvs[valid_masks_pred_sq == 0] = ignore_index

            loss_x = criterion_l(pred_x, mask_x)

            mask_u_w_cutmixed = mask_u_w.clone()
            conf_u_w_cutmixed = conf_u_w.clone()
            ignore_mask_cutmixed = ignore_mask.clone()
            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w.flip(0)[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w.flip(0)[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask.flip(0)[cutmix_box == 1]

            loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed)
            loss_u_s = confidence_weighted_loss(
                loss_u_s,
                conf_u_w_cutmixed,
                ignore_mask_cutmixed,
                ignore_index,
                conf_thresh=conf_thresh,
            )

            loss_u_rvs = criterion_u(pred_recovered, mask_u_w_rvs)
            loss_u_rvs = confidence_weighted_loss(
                loss_u_rvs,
                conf_u_w,
                ignore_mask_rvs,
                ignore_index,
                conf_thresh=conf_thresh,
            )

            loss_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_fp_mask = (conf_u_w >= conf_thresh) & (ignore_mask != ignore_index)
            loss_fp = (loss_fp * loss_fp_mask).sum() / loss_fp_mask.sum().clamp(min=1.0)

            loss = (loss_x + loss_u_s * 0.25 + loss_u_rvs * 0.25 + loss_fp * 0.5) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if is_main and viz is not None and i < 10:
                viz.push(
                    {
                        "img_x": (img_x[0], Visualizer.TENSOR),
                        "mask_x": (mask_x[0], Visualizer.SEGMENTATION),
                        "pred_x": (pred_x.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                        "img_u_c": (img_u_c[0], Visualizer.TENSOR),
                        "pred_u_rvs": (
                            pred_u_rvs.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_recovered": (
                            pred_recovered.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "img_u_w": (img_u_w[0], Visualizer.TENSOR),
                        "mask_u_w": (mask_u_w[0], Visualizer.SEGMENTATION),
                        "img_u_s": (img_u_s[0], Visualizer.TENSOR),
                        "pred_u_s": (
                            pred_u_s.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_u_w_fp": (
                            pred_u_w_fp.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "mask_u_w_rvs": (
                            mask_u_w_rvs[0],
                            Visualizer.SEGMENTATION,
                        ),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            if is_main and i % 100 == 0:
                with torch.no_grad():
                    valid_ratio = valid_masks_pred_sq.float().mean().item()
                    conf_in_valid = conf_u_w[valid_masks_pred_sq > 0]
                    high_conf_ratio = (
                        (conf_in_valid >= conf_thresh).float().mean().item()
                        if conf_in_valid.numel() > 0
                        else 0.0
                    )
                    agree_ratio = (
                        (pred_recovered.argmax(1) == mask_u_w)[valid_masks_pred_sq > 0]
                        .float()
                        .mean()
                        .item()
                        if (valid_masks_pred_sq > 0).any()
                        else 0.0
                    )
                    logger.info(
                        f"[RVS Debug] valid_ratio={valid_ratio:.3f}, "
                        f"high_conf_valid={high_conf_ratio:.3f}, "
                        f"student-teacher agree={agree_ratio:.3f}"
                    )

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(
                    param_ema * ema_ratio + param.detach() * (1 - ema_ratio)
                )
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(
                    buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio)
                )

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_u_s.update(loss_u_s.item())
            total_loss_u_rvs.update(loss_u_rvs.item())
            total_loss_fp.update(loss_fp.item())

            valid_unsup = (ignore_mask != ignore_index).sum().item()
            mask_ratio = (
                ((conf_u_w >= conf_thresh) & (ignore_mask != ignore_index)).sum().item()
                / max(valid_unsup, 1)
            )
            total_mask_ratio.update(mask_ratio)

            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if is_main and writer is not None:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_u_s", loss_u_s.item(), iters)
                writer.add_scalar("train/loss_u_rvs", loss_u_rvs.item(), iters)
                writer.add_scalar("train/loss_fp", loss_fp.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            log_freq = max(len(trainloader_u) // 8, 1)
            if is_main and (i % log_freq == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, "
                    "Loss u_s: {:.3f}, Loss u_rvs: {:.3f}, Loss fp: {:.3f}, "
                    "Mask ratio: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_u_s.avg,
                        total_loss_u_rvs.avg,
                        total_loss_fp.avg,
                        total_mask_ratio.avg,
                    )
                )

        mIoU, iou_class = validation_distributed(
            cfg, model, valloader, eval_mode, ignore_index
        )
        mIoU_ema, iou_class_ema = validation_distributed(
            cfg, model_ema, valloader, eval_mode, ignore_index
        )

        is_best = mIoU >= previous_best
        is_best_ema = mIoU_ema >= previous_best_ema

        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if is_best:
            best_epoch = epoch
        if is_best_ema:
            best_epoch_ema = epoch

        if is_main:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ({}) ***** >>>> Class [{:} {:}] IoU: {:.2f}, "
                    "EMA: {:.2f}".format(
                        eval_mode,
                        cls_idx,
                        CLASSES[cfg["dataset"]][cls_idx],
                        iou,
                        iou_class_ema[cls_idx],
                    )
                )
            logger.info(
                "***** Evaluation ({}) ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n".format(
                    eval_mode, mIoU, mIoU_ema
                )
            )

            if writer is not None:
                writer.add_scalar("eval/mIoU", mIoU, epoch)
                writer.add_scalar("eval/mIoU_ema", mIoU_ema, epoch)
                for cls_idx, iou in enumerate(iou_class):
                    writer.add_scalar(
                        f'eval/{CLASSES[cfg["dataset"]][cls_idx]}_IoU', iou, epoch
                    )
                    writer.add_scalar(
                        f'eval/{CLASSES[cfg["dataset"]][cls_idx]}_IoU_ema',
                        iou_class_ema[cls_idx],
                        epoch,
                    )

            checkpoint = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
                "previous_best_ema": previous_best_ema,
                "best_epoch": best_epoch,
                "best_epoch_ema": best_epoch_ema,
                "resolved_ignore_index": ignore_index,
                "resolved_eval_mode": eval_mode,
            }
            torch.save(checkpoint, latest_path)
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))
            if is_best_ema:
                torch.save(checkpoint, os.path.join(args.save_path, "best_ema.pth"))


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
