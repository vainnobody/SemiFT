import argparse
import logging
import os
import pprint

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.semi_rs import SemiDataset
from supervised import validation_cpu
from util.classes import CLASSES
from util.utils import AverageMeter
from util.ssl_method_utils import (
    build_criterions,
    build_logger_and_runtime,
    build_model,
    build_optimizer,
    get_backbone_info,
    log_model_info,
    maybe_load_checkpoint,
    save_checkpoint,
    update_lr,
    wrap_ddp,
)
from util.viz import Visualizer
from util.wscl_utils import (
    entropy_map,
    generate_unsup_aug_dc,
    generate_unsup_aug_ds,
    generate_unsup_aug_sc,
    generate_unsup_aug_sdc,
)
from dataset.val import ValDataset


AUG_HANDLERS = {
    "SC": lambda conf, mask, s1, s2: generate_unsup_aug_sc(conf, mask, s1),
    "DS": lambda conf, mask, s1, s2: (conf, mask, generate_unsup_aug_ds(s1, s2)),
    "DC": generate_unsup_aug_dc,
    "SDC": generate_unsup_aug_sdc,
}


def get_parser():
    parser = argparse.ArgumentParser(description="WSCL with DPT/DINO backbone")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def build_dataloaders(args, cfg):
    trainset_u = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        ignore_index=cfg["ignore_index"],
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
    valset = ValDataset(cfg["dataset"], cfg["data_root"], "val", ignore_value=cfg["ignore_index"])
    workers = cfg.get("workers", 4)
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg["batch_size"], sampler=trainsampler_l, pin_memory=True, num_workers=workers, drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg["batch_size"], sampler=trainsampler_u, pin_memory=True, num_workers=workers, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, sampler=valsampler, pin_memory=True, num_workers=1, drop_last=False)
    return trainloader_l, trainloader_u, valloader


def main(args, cfg):
    logger, rank, world_size, writer = build_logger_and_runtime(args, cfg)
    model, load_result = build_model(cfg, method="fixmatch")
    optimizer = build_optimizer(cfg, model)
    log_model_info(logger, rank, model, load_result)
    model, local_rank = wrap_ddp(model)
    criterion_l, criterion_u = build_criterions(cfg, local_rank)
    trainloader_l, trainloader_u, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader_u) * cfg["epochs"]

    state = maybe_load_checkpoint(args, model, optimizer)
    previous_best = state["previous_best"]
    best_epoch = state["best_epoch"]
    start_epoch = state["epoch"]

    wscl_cfg = cfg.get("wscl", {})
    aug_mode = wscl_cfg.get("aug_mode", "SDC")
    low_entropy_percent = wscl_cfg.get("percent", 20)
    unsup_weight = wscl_cfg.get("loss_weight", 1.0)

    filename = None
    if rank == 0:
        from datetime import datetime
        filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz = Visualizer(save_dir=f"./viz/{filename}", dataset=cfg["dataset"])
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
            img_u_w, img_u_s1, img_u_s2, ignore_mask, _, _ = batch_u
            img_u_w = img_u_w.cuda()
            img_u_s1 = img_u_s1.cuda()
            img_u_s2 = img_u_s2.cuda()
            ignore_mask = ignore_mask.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_w = model(img_u_w).detach()
                prob_u_w = pred_u_w.softmax(dim=1)
                conf_u_w, mask_u_w = prob_u_w.max(dim=1)

            if aug_mode not in AUG_HANDLERS:
                raise ValueError(f"Unknown wscl aug_mode: {aug_mode}")
            conf_mix, mask_mix, img_u_s = AUG_HANDLERS[aug_mode](conf_u_w, mask_u_w, img_u_s1, img_u_s2)

            model.train()
            pred_all = model(torch.cat((img_x, img_u_s)))
            pred_x, pred_u_s = pred_all.split([img_x.shape[0], img_u_s.shape[0]])

            loss_x = criterion_l(pred_x, mask_x)
            loss_u = criterion_u(pred_u_s, mask_mix)
            entropy_u = entropy_map(pred_u_s.softmax(dim=1), 1)
            entropy_valid = entropy_u[ignore_mask != 255]
            if entropy_valid.numel() > 0:
                threshold = np.percentile(entropy_valid.detach().cpu().numpy().flatten(), low_entropy_percent)
                mask_valid = (entropy_u <= threshold) & (ignore_mask != 255)
                loss_u = (loss_u * mask_valid).sum() / mask_valid.sum().clamp(min=1).float()
                mask_ratio = mask_valid.sum().float() / (ignore_mask != 255).sum().clamp(min=1).float()
            else:
                loss_u = loss_u.sum() * 0.0
                mask_ratio = torch.tensor(0.0, device=loss_u.device)

            loss = loss_x + unsup_weight * loss_u
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u.item())
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = update_lr(optimizer, cfg, iters, total_iters)
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
                        "mask_u": (mask_mix[0], Visualizer.SEGMENTATION),
                        "pred_u": (pred_u_s.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                    })
                    viz.render(f"epoch_{epoch}_iter_{i}")
                    viz.reset()
                if i % max(1, len(trainloader_u) // 8) == 0:
                    logger.info(
                        "Iters: %d, LR: %.7f, Total loss: %.3f, Loss x: %.3f, Loss s: %.3f, Mask ratio: %.3f",
                        i, lr, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_mask_ratio.avg,
                    )

        mIoU, iou_class = validation_cpu(cfg, model, valloader)
        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info("***** Evaluation ***** >>>> Class [%d %s] IoU: %.2f", cls_idx, CLASSES[cfg["dataset"]][cls_idx], iou)
            logger.info("***** Evaluation ***** >>>> MeanIoU: %.2f\n", mIoU)
            writer.add_scalar("eval/mIoU", mIoU, epoch)

        is_best = mIoU >= previous_best
        previous_best = max(mIoU, previous_best)
        if mIoU == previous_best:
            best_epoch = epoch
        save_checkpoint(args, rank, model, optimizer, epoch, previous_best, best_epoch, is_best=is_best)


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
