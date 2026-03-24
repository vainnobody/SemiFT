import argparse
import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from dataset.semi_ranpaste import SemiDataset
from dataset.val import ValDataset
from util.classes import CLASSES
from util.ranpaste_utils import (
    build_ranpaste_images,
    build_ranpaste_targets,
)
from util.ssl_method_utils import (
    build_criterions,
    build_logger_and_runtime,
    build_model,
    build_optimizer,
    get_local_rank,
    log_model_info,
    maybe_load_checkpoint,
    save_checkpoint,
    update_lr,
    wrap_ddp,
)
from util.utils import AverageMeter
from util.validation import validation_cpu as shared_validation_cpu
from util.viz import Visualizer


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


@torch.no_grad()
def forward_pseudo_labels(model, img_u_w):
    pred_u_w = model(img_u_w).detach()
    conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
    mask_u_w = pred_u_w.argmax(dim=1)
    return pred_u_w, conf_u_w, mask_u_w


@torch.no_grad()
def build_pasted_batches(
    img_u_w,
    img_u_s,
    img_x,
    mask_x,
    ignore_mask,
    paste_mask,
    conf_thresh,
    pseudo_model,
    ignore_index,
):
    _, conf_u_w, mask_u_w = forward_pseudo_labels(pseudo_model, img_u_w)
    img_u_w_mix = build_ranpaste_images(img_u_w, img_x, paste_mask)
    img_u_s_mix = build_ranpaste_images(img_u_s, img_x, paste_mask)
    target_mix, valid_mask = build_ranpaste_targets(
        mask_u_w,
        conf_u_w,
        ignore_mask,
        mask_x,
        paste_mask,
        conf_thresh,
        ignore_index=ignore_index,
    )
    return img_u_w_mix, img_u_s_mix, target_mix, valid_mask, conf_u_w


def get_parser():
    parser = argparse.ArgumentParser(
        description="RanPaste training rebuilt on top of the UniMatch V2-style scaffold"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def build_dataloaders(args, cfg):
    paste_cfg = cfg.get("paste", {})
    trainset_u = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        ignore_index=cfg["ignore_index"],
        paste_cfg=paste_cfg,
    )
    trainset_l = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(trainset_u.ids),
        ignore_index=cfg["ignore_index"],
    )
    valset = ValDataset(
        cfg["dataset"], cfg["data_root"], "val", ignore_value=cfg["ignore_index"]
    )

    workers = cfg.get("workers", 4)
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
        sampler=trainsampler_l,
    )

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
        sampler=trainsampler_u,
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        drop_last=False,
        sampler=valsampler,
    )
    return trainloader_l, trainloader_u, valloader


def main(args, cfg):
    logger, rank, _, writer = build_logger_and_runtime(args, cfg)

    model, load_result = build_model(cfg, method="ranpaste")
    optimizer = build_optimizer(cfg, model)
    log_model_info(logger, rank, model, load_result=load_result)

    model, local_rank = wrap_ddp(model, logger=logger, rank=rank, save_path=args.save_path)
    criterion_l, _ = build_criterions(cfg, local_rank)
    criterion_u = nn.CrossEntropyLoss(
        reduction="none", ignore_index=cfg["ignore_index"]
    ).cuda(local_rank)

    trainloader_l, trainloader_u, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader_u) * cfg["epochs"]

    state = maybe_load_checkpoint(args, model, optimizer, logger=logger, rank=rank)
    previous_best = state["previous_best"]
    best_epoch = state["best_epoch"]
    start_epoch = state["epoch"] + 1

    from datetime import datetime

    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz = Visualizer(save_dir=f"./viz/{filename}", dataset=cfg["dataset"])
    conf_thresh = cfg["conf_thresh"]

    for epoch in range(start_epoch, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}".format(
                    epoch, previous_best, best_epoch
                )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_u = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        loader = zip(trainloader_l, trainloader_u)

        model.train()

        for i, ((img_x, mask_x), (img_u_w, img_u_s, ignore_mask, paste_mask)) in enumerate(loader):
            img_x, mask_x = img_x.cuda(local_rank), mask_x.cuda(local_rank)
            img_u_w = img_u_w.cuda(local_rank)
            img_u_s = img_u_s.cuda(local_rank)
            ignore_mask = ignore_mask.cuda(local_rank)
            paste_mask = paste_mask.cuda(local_rank)

            with torch.no_grad():
                (
                    img_u_w_mix,
                    img_u_s_mix,
                    target_mix,
                    valid_mask,
                    conf_u_w,
                ) = build_pasted_batches(
                    img_u_w,
                    img_u_s,
                    img_x,
                    mask_x,
                    ignore_mask,
                    paste_mask,
                    conf_thresh,
                    model,
                    cfg["ignore_index"],
                )

            num_lb, num_ulb = img_x.shape[0], img_u_s_mix.shape[0]
            pred_x, pred_u = model(torch.cat((img_x, img_u_s_mix))).split([num_lb, num_ulb])

            loss_x = criterion_l(pred_x, mask_x)
            loss_u = criterion_u(pred_u, target_mix)
            loss_u = (loss_u * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1.0)
            unsup_weight = cfg.get("ranpaste", {}).get("unsup_weight", 1.0)
            loss = (loss_x + unsup_weight * loss_u) / (1.0 + unsup_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_u.update(loss_u.item())
            mask_ratio = valid_mask.float().mean()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = update_lr(optimizer, cfg, iters, total_iters)

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_u", loss_u.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio.item(), iters)
                writer.add_scalar("train/lr", lr, iters)

            if i < 10:
                viz.push(
                    {
                        "img_x": (img_x[0], Visualizer.TENSOR),
                        "mask_x": (mask_x[0], Visualizer.SEGMENTATION),
                        "img_u_w_mix": (img_u_w_mix[0], Visualizer.TENSOR),
                        "img_u_s_mix": (img_u_s_mix[0], Visualizer.TENSOR),
                        "target_mix": (target_mix[0], Visualizer.SEGMENTATION),
                        "pred_u": (pred_u.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            log_interval = max(1, len(trainloader_u) // 8)
            if rank == 0 and i % log_interval == 0:
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss u: {:.3f}, Mask ratio: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_u.avg,
                        total_mask_ratio.avg,
                    )
                )

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode", "slide_window" if cfg["dataset"] == "cityscapes" else "original"
        )
        val_cfg.setdefault("ignore_index", cfg.get("ignore_index", 255))
        eval_mode = val_cfg["eval_mode"]
        mIoU, iou_class = validation_cpu(val_cfg, model, valloader)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}".format(
                        cls_idx, CLASSES[cfg["dataset"]][cls_idx], iou
                    )
                )
            logger.info(
                "***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n".format(
                    eval_mode, mIoU
                )
            )
            writer.add_scalar("eval/mIoU", mIoU, epoch)
            for cls_idx, iou in enumerate(iou_class):
                writer.add_scalar(
                    "eval/%s_IoU" % (CLASSES[cfg["dataset"]][cls_idx]), iou, epoch
                )

        is_best = mIoU >= previous_best
        previous_best = max(mIoU, previous_best)
        if mIoU == previous_best:
            best_epoch = epoch

        save_checkpoint(
            args,
            rank,
            model,
            optimizer,
            epoch,
            previous_best,
            best_epoch,
            is_best=is_best,
        )


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
