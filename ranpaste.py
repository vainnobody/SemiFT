import argparse
import os
import random

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.semi_rs import SemiDataset
from dataset.val import ValDataset
from util.classes import CLASSES
from util.ssl_method_utils import (
    build_criterions,
    build_ema_model,
    build_logger_and_runtime,
    build_model,
    build_optimizer,
    log_model_info,
    maybe_load_checkpoint,
    save_checkpoint,
    update_ema,
    update_lr,
    wrap_ddp,
)
from util.utils import AverageMeter
from util.validation import validation_cpu as shared_validation_cpu
from util.viz import Visualizer


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def get_parser():
    parser = argparse.ArgumentParser(description="RanPaste with DPT/DINO backbone")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def build_dataloaders(args, cfg):
    trainset_u = SemiDataset(
        cfg["dataset"], cfg["data_root"], "train_u", cfg["crop_size"], args.unlabeled_id_path, ignore_index=cfg["ignore_index"]
    )
    trainset_l = SemiDataset(
        cfg["dataset"], cfg["data_root"], "train_l", cfg["crop_size"], args.labeled_id_path, nsample=len(trainset_u.ids), ignore_index=cfg["ignore_index"]
    )
    valset = ValDataset(cfg["dataset"], cfg["data_root"], "val", ignore_value=cfg["ignore_index"])
    workers = cfg.get("workers", 4)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg["batch_size"], sampler=torch.utils.data.distributed.DistributedSampler(trainset_l), pin_memory=True, num_workers=workers, drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg["batch_size"], sampler=torch.utils.data.distributed.DistributedSampler(trainset_u), pin_memory=True, num_workers=workers, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, sampler=torch.utils.data.distributed.DistributedSampler(valset), pin_memory=True, num_workers=1, drop_last=False)
    return trainloader_l, trainloader_u, valloader


def sample_paste_boxes(batch_size, h, w, ratio=0.5, device=None):
    paste_h = max(1, int(h * ratio))
    paste_w = max(1, int(w * ratio))
    boxes = torch.zeros((batch_size, h, w), dtype=torch.bool, device=device)
    max_y = max(h - paste_h, 0)
    max_x = max(w - paste_w, 0)
    for i in range(batch_size):
        y = random.randint(0, max_y) if max_y > 0 else 0
        x = random.randint(0, max_x) if max_x > 0 else 0
        boxes[i, y : y + paste_h, x : x + paste_w] = True
    return boxes


def main(args, cfg):
    logger, rank, world_size, writer = build_logger_and_runtime(args, cfg)
    model, load_result = build_model(cfg, method="fixmatch")
    optimizer = build_optimizer(cfg, model)
    log_model_info(logger, rank, model, load_result)
    model, local_rank = wrap_ddp(model, logger=logger, rank=rank, save_path=args.save_path)
    model_ema = build_ema_model(model)
    criterion_l, criterion_u = build_criterions(cfg, local_rank)
    trainloader_l, trainloader_u, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader_u) * cfg["epochs"]

    state = maybe_load_checkpoint(args, model, optimizer, model_ema=model_ema, logger=logger, rank=rank)
    previous_best = state["previous_best"]
    previous_best_ema = state["previous_best_ema"]
    best_epoch = state["best_epoch"]
    best_epoch_ema = state["best_epoch_ema"]
    start_epoch = state["epoch"]

    ran_cfg = cfg.get("ranpaste", {})
    pseudo_thresh = ran_cfg.get("pseudo_thresh", 0.6)
    sup_conf_thresh = ran_cfg.get("sup_conf_thresh", 0.9)
    paste_ratio = ran_cfg.get("paste_ratio", 0.5)
    unsup_weight = ran_cfg.get("loss_weight", 1.0)

    if rank == 0:
        from datetime import datetime
        viz = Visualizer(save_dir=f"./viz/{datetime.now().strftime('%Y%m%d_%H%M%S')}", dataset=cfg["dataset"])
    else:
        viz = None

    for epoch in range(start_epoch + 1, cfg["epochs"]):
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        model.train()
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        for i, ((img_x, mask_x), batch_u) in enumerate(zip(trainloader_l, trainloader_u)):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2 = batch_u
            img_u_w = img_u_w.cuda()
            img_u_s1 = img_u_s1.cuda()
            ignore_mask = ignore_mask.cuda()

            with torch.no_grad():
                pred_x_for_gate = model(img_x).detach()
                model_ema.eval()
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            paste_box = sample_paste_boxes(img_u_s1.shape[0], img_u_s1.shape[-2], img_u_s1.shape[-1], paste_ratio, img_u_s1.device)
            img_u_s = img_u_s1.clone()
            pseudo_label = mask_u_w.clone()
            pseudo_conf = conf_u_w.clone()
            for b in range(img_u_s.shape[0]):
                src_idx = b % img_x.shape[0]
                img_u_s[b, :, paste_box[b]] = img_x[src_idx, :, paste_box[b]]
                pseudo_label[b, paste_box[b]] = mask_x[src_idx, paste_box[b]]
                pseudo_conf[b, paste_box[b]] = 1.0
                ignore_mask[b, paste_box[b]] = 0

            pred_x = model(img_x)
            pred_u_s = model(img_u_s)

            if sup_conf_thresh is not None:
                pred_soft = pred_x.softmax(dim=1).max(dim=1)[0]
                sup_gate = (pred_soft < sup_conf_thresh).float()
                loss_x_map = criterion_u(pred_x, mask_x)
                loss_x = (loss_x_map * sup_gate).sum() / sup_gate.sum().clamp(min=1.0)
            else:
                loss_x = criterion_l(pred_x, mask_x)

            loss_u = criterion_u(pred_u_s, pseudo_label)
            valid_mask = (pseudo_conf >= pseudo_thresh) & (ignore_mask != 255)
            loss_u = (loss_u * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
            mask_ratio = valid_mask.sum().float() / (ignore_mask != 255).sum().clamp(min=1.0).float()

            loss = loss_x + unsup_weight * loss_u
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader_u) + i
            lr = update_lr(optimizer, cfg, iters, total_iters)
            update_ema(model, model_ema, iters, max_decay=ran_cfg.get("ema_decay", 0.999))

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u.item())
            total_mask_ratio.update(mask_ratio.item())

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_s", loss_u.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio.item(), iters)
                if viz is not None and i < 5:
                    viz.push({
                        "img_x": (img_x[0], Visualizer.TENSOR),
                        "mask_x": (mask_x[0], Visualizer.SEGMENTATION),
                        "pred_x": (pred_x.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                        "img_u_s": (img_u_s[0], Visualizer.TENSOR),
                        "pseudo_label": (pseudo_label[0], Visualizer.SEGMENTATION),
                        "pred_u": (pred_u_s.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                    })
                    viz.render(f"epoch_{epoch}_iter_{i}")
                    viz.reset()
                if i % max(1, len(trainloader_u) // 8) == 0:
                    logger.info(
                        "Iters: %d, LR: %.7f, Total loss: %.3f, Loss x: %.3f, Loss s: %.3f, Mask ratio: %.3f",
                        i, lr, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_mask_ratio.avg,
                    )

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode", "slide_window" if cfg["dataset"] == "cityscapes" else "original"
        )
        val_cfg.setdefault("ignore_index", cfg.get("ignore_index", 255))
        mIoU, iou_class = validation_cpu(val_cfg, model, valloader)
        mIoU_ema, iou_class_ema = validation_cpu(val_cfg, model_ema, valloader)
        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [%d %s] IoU: %.2f, EMA: %.2f",
                    cls_idx,
                    CLASSES[cfg["dataset"]][cls_idx],
                    iou,
                    iou_class_ema[cls_idx],
                )
            logger.info("***** Evaluation ***** >>>> MeanIoU: %.2f, EMA: %.2f\n", mIoU, mIoU_ema)
            writer.add_scalar("eval/mIoU", mIoU, epoch)
            writer.add_scalar("eval/mIoU_ema", mIoU_ema, epoch)

        is_best = mIoU >= previous_best
        previous_best = max(previous_best, mIoU)
        previous_best_ema = max(previous_best_ema, mIoU_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        save_checkpoint(
            args,
            rank,
            model,
            optimizer,
            epoch,
            previous_best,
            best_epoch,
            model_ema=model_ema,
            previous_best_ema=previous_best_ema,
            best_epoch_ema=best_epoch_ema,
            is_best=is_best,
        )


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
