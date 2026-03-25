import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import yaml

from dataset.semi import SemiDataset as PascalSemiDataset
from dataset.semi_rs import SemiDataset as RemoteSemiDataset
from dataset.val import ValDataset
from util.classes import CLASSES
from util.ssl_method_utils import (
    build_criterions,
    build_logger_and_runtime,
    build_model,
    build_optimizer,
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


def get_parser():
    parser = argparse.ArgumentParser(
        description="UniMatch V1 rebuilt on top of the SemiFT training scaffold"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def resolve_unimatch_dataset_cls(cfg):
    dataset_variant = (
        cfg.get("unimatch", {}).get("dataset_variant")
        or cfg.get("semi_dataset_variant")
        or "semi_rs"
    )
    if dataset_variant == "semi":
        return PascalSemiDataset
    if dataset_variant == "semi_rs":
        return RemoteSemiDataset
    raise ValueError(
        f"Unsupported UniMatch dataset_variant '{dataset_variant}'. "
        "Expected 'semi' or 'semi_rs'."
    )


def build_dataloaders(args, cfg):
    dataset_cls = resolve_unimatch_dataset_cls(cfg)
    trainset_u = dataset_cls(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        ignore_index=cfg["ignore_index"],
    )
    trainset_l = dataset_cls(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(trainset_u.ids),
        ignore_index=cfg["ignore_index"],
    )
    trainset_u_mix = dataset_cls(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        ignore_index=cfg["ignore_index"],
    )
    valset = ValDataset(
        cfg["dataset"], cfg["data_root"], "val", ignore_value=cfg["ignore_index"]
    )

    workers = cfg.get("workers", 4)
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainsampler_u_mix = torch.utils.data.distributed.DistributedSampler(trainset_u_mix)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)

    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
        sampler=trainsampler_l,
    )
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
        sampler=trainsampler_u,
    )
    trainloader_u_mix = DataLoader(
        trainset_u_mix,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
        sampler=trainsampler_u_mix,
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        drop_last=False,
        sampler=valsampler,
    )
    return trainloader_l, trainloader_u, trainloader_u_mix, valloader


def main(args, cfg):
    logger, rank, _, writer = build_logger_and_runtime(args, cfg)
    model, load_result = build_model(cfg, method="unimatch")
    optimizer = build_optimizer(cfg, model)
    log_model_info(logger, rank, model, load_result=load_result)
    model, local_rank = wrap_ddp(model, logger=logger, rank=rank, save_path=args.save_path)
    criterion_l, criterion_u = build_criterions(cfg, local_rank)

    trainloader_l, trainloader_u, trainloader_u_mix, valloader = build_dataloaders(
        args, cfg
    )
    total_iters = len(trainloader_u) * cfg["epochs"]

    state = maybe_load_checkpoint(args, model, optimizer, logger=logger, rank=rank)
    previous_best = state["previous_best"]
    best_epoch = state["best_epoch"]
    start_epoch = state["epoch"] + 1

    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz = Visualizer(save_dir=f"./viz/{filename}", dataset=cfg["dataset"])

    for epoch in range(start_epoch, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}".format(
                    epoch, previous_best, best_epoch
                )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        trainloader_u_mix.sampler.set_epoch(epoch)
        model.train()

        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
            (
                img_u_w_mix,
                img_u_s1_mix,
                img_u_s2_mix,
                ignore_mask_mix,
                _,
                _,
            ),
        ) in enumerate(loader):
            img_x = img_x.cuda(local_rank)
            mask_x = mask_x.cuda(local_rank)
            img_u_w = img_u_w.cuda(local_rank)
            img_u_s1 = img_u_s1.cuda(local_rank)
            img_u_s2 = img_u_s2.cuda(local_rank)
            ignore_mask = ignore_mask.cuda(local_rank)
            cutmix_box1 = cutmix_box1.cuda(local_rank)
            cutmix_box2 = cutmix_box2.cuda(local_rank)
            img_u_w_mix = img_u_w_mix.cuda(local_rank)
            img_u_s1_mix = img_u_s1_mix.cuda(local_rank)
            img_u_s2_mix = img_u_s2_mix.cuda(local_rank)
            ignore_mask_mix = ignore_mask_mix.cuda(local_rank)

            with torch.no_grad():
                was_training = model.training
                model.eval()
                try:
                    pred_u_w_mix = model(img_u_w_mix).detach()
                finally:
                    if was_training:
                        model.train()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = (
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            )
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = (
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            )

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            preds, preds_fp = model(torch.cat((img_x, img_u_w)), need_fp=True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1 = mask_u_w.clone()
            conf_u_w_cutmixed1 = conf_u_w.clone()
            ignore_mask_cutmixed1 = ignore_mask.clone()
            mask_u_w_cutmixed2 = mask_u_w.clone()
            conf_u_w_cutmixed2 = conf_u_w.clone()
            ignore_mask_cutmixed2 = ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * (
                (conf_u_w_cutmixed1 >= cfg["conf_thresh"])
                & (ignore_mask_cutmixed1 != 255)
            )
            loss_u_s1 = loss_u_s1.sum() / (
                (ignore_mask_cutmixed1 != 255).sum().clamp(min=1).item()
            )

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * (
                (conf_u_w_cutmixed2 >= cfg["conf_thresh"])
                & (ignore_mask_cutmixed2 != 255)
            )
            loss_u_s2 = loss_u_s2.sum() / (
                (ignore_mask_cutmixed2 != 255).sum().clamp(min=1).item()
            )

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * (
                (conf_u_w >= cfg["conf_thresh"]) & (ignore_mask != 255)
            )
            loss_u_w_fp = loss_u_w_fp.sum() / (
                (ignore_mask != 255).sum().clamp(min=1).item()
            )

            loss = (
                loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5
            ) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i < 10:
                viz.push(
                    {
                        "img_x": (img_x[0], Visualizer.TENSOR),
                        "mask_x": (mask_x[0], Visualizer.SEGMENTATION),
                        "pred_x": (pred_x.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                        "img_u_s1": (img_u_s1[0], Visualizer.TENSOR),
                        "img_u_s2": (img_u_s2[0], Visualizer.TENSOR),
                        "mask_u_w_cutmixed1": (
                            mask_u_w_cutmixed1[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "mask_u_w_cutmixed2": (
                            mask_u_w_cutmixed2[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_u_s1": (
                            pred_u_s1.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_u_s2": (
                            pred_u_s2.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_u_w_fp": (
                            pred_u_w_fp.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            mask_ratio = (
                ((conf_u_w >= cfg["conf_thresh"]) & (ignore_mask != 255)).sum().item()
                / (ignore_mask != 255).sum().clamp(min=1).item()
            )
            total_mask_ratio.update(mask_ratio)

            iters = epoch * len(trainloader_u) + i
            update_lr(optimizer, cfg, iters, total_iters)

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar(
                    "train/loss_s", (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters
                )
                writer.add_scalar("train/loss_w_fp", loss_u_w_fp.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            if (i % max(1, len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, "
                    "Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_loss_w_fp.avg,
                        total_mask_ratio.avg,
                    )
                )

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode",
            "sliding_window" if cfg["dataset"] == "cityscapes" else "original",
        )
        val_cfg.setdefault("ignore_index", cfg.get("ignore_index", 255))
        eval_mode = val_cfg["eval_mode"]
        mIoU, iou_class = validation_cpu(val_cfg, model, valloader)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}".format(
                        cls_idx,
                        CLASSES[cfg["dataset"]][cls_idx],
                        iou,
                    )
                )
            logger.info(
                "***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n".format(
                    eval_mode, mIoU
                )
            )
            writer.add_scalar("eval/mIoU", mIoU, epoch)
            for idx, iou in enumerate(iou_class):
                writer.add_scalar(
                    f"eval/{CLASSES[cfg['dataset']][idx]}_IoU", iou, epoch
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
