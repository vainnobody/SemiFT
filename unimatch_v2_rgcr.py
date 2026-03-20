import argparse
from copy import deepcopy
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi_rgcr import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.upernet import UperNet
from model.semseg.rgcr_utils import scale_back
from supervised import validation_cpu
from util.classes import CLASSES
from util.dist_helper import setup_distributed
from util.focal import FocalLoss
from util.ohem import ProbOhemCrossEntropy2d
from util.train_utils import confidence_weighted_loss
from util.utils import AverageMeter, count_params, init_log
from util.viz import Visualizer


def get_parser():
    parser = argparse.ArgumentParser(
        description="UniMatch-v2 with RGCR-style geometric consistency"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


@torch.no_grad()
def apply_cutmix_to_map(map_tensor, cutmix_box):
    cutmixed = map_tensor.clone()
    cutmixed[cutmix_box == 1] = map_tensor.flip(0)[cutmix_box == 1]
    return cutmixed


@torch.no_grad()
def binarize_valid_mask(mask_tensor, threshold=0.5):
    return (mask_tensor >= threshold).float()


@torch.no_grad()
def validate_target_range(name, target, nclass, ignore_index, logger=None, rank=0):
    invalid = (target != ignore_index) & ((target < 0) | (target >= nclass))
    if invalid.any():
        bad_vals = torch.unique(target[invalid]).detach().cpu().tolist()
        msg = (
            f"[TargetRangeError] {name}: invalid target values {bad_vals}, "
            f"min={target.min().item()}, max={target.max().item()}, "
            f"ignore_index={ignore_index}, nclass={nclass}"
        )
        if logger is not None and rank == 0:
            logger.error(msg)
        raise ValueError(msg)


def main(args, cfg):
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

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

    if cfg["model"] == "dpt":
        model = DPT(
            **{**model_configs[backbone_size], "nclass": cfg["nclass"]},
            backbone_version=backbone_version,
        )
    elif cfg["model"] == "upernet":
        model = UperNet(
            **{**model_configs[backbone_size], "nclass": cfg["nclass"]},
            backbone_version=backbone_version,
        )
    else:
        raise NotImplementedError(f"Unknown model: {cfg['model']}")

    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth', map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(state_dict)

    if cfg["lock_backbone"]:
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

    if rank == 0:
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
            "%s criterion is not implemented" % cfg["criterion"]["name"]
        )

    ignore_index = cfg["ignore_index"]
    conf_thresh = cfg["conf_thresh"]
    criterion_u = nn.CrossEntropyLoss(
        reduction="none", ignore_index=ignore_index
    ).cuda(local_rank)

    trainset_u = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        ignore_index=ignore_index,
    )
    trainset_l = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(trainset_u.ids),
        ignore_index=ignore_index,
    )
    valset = ValDataset(
        cfg["dataset"], cfg["data_root"], "val", ignore_value=ignore_index
    )

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_l,
    )

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=4,
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

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        model_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        previous_best_ema = checkpoint.get("previous_best_ema", 0.0)
        best_epoch = checkpoint.get("best_epoch", 0)
        best_epoch_ema = checkpoint.get("best_epoch_ema", 0)

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    from datetime import datetime

    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz = Visualizer(save_dir=f"./viz/{filename}", dataset=cfg["dataset"])

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, "
                "EMA: {:.2f} @epoch-{:}".format(
                    epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema
                )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s1 = AverageMeter()
        total_loss_s2 = AverageMeter()
        total_loss_rvs = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_rvs_valid_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        model.train()

        for i, (
            (img_x, mask_x),
            (
                img_u_w,
                img_u_s1,
                img_u_s2,
                img_u_rvs,
                ignore_mask,
                cutmix_box1,
                cutmix_box2,
                box,
                mask_c,
            ),
        ) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2, img_u_rvs = (
                img_u_w.cuda(),
                img_u_s1.cuda(),
                img_u_s2.cuda(),
                img_u_rvs.cuda(),
            )
            ignore_mask, cutmix_box1, cutmix_box2 = (
                ignore_mask.cuda(),
                cutmix_box1.cuda(),
                cutmix_box2.cuda(),
            )
            box, mask_c = box.cuda(), mask_c.cuda()

            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = (
                img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            )
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = (
                img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            )

            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(
                torch.cat((img_u_s1, img_u_s2)), comp_drop=True
            ).chunk(2)
            pred_u_rvs = model(img_u_rvs)
            pred_recovered, valid_masks_pred = scale_back(
                pred_u_rvs, mask_c, cfg["crop_size"], box
            )
            valid_masks_pred_sq = binarize_valid_mask(
                valid_masks_pred.squeeze(1).float()
            )

            mask_u_w_cutmixed1 = apply_cutmix_to_map(mask_u_w, cutmix_box1)
            conf_u_w_cutmixed1 = apply_cutmix_to_map(conf_u_w, cutmix_box1)
            ignore_mask_cutmixed1 = apply_cutmix_to_map(ignore_mask, cutmix_box1)

            mask_u_w_cutmixed2 = apply_cutmix_to_map(mask_u_w, cutmix_box2)
            conf_u_w_cutmixed2 = apply_cutmix_to_map(conf_u_w, cutmix_box2)
            ignore_mask_cutmixed2 = apply_cutmix_to_map(ignore_mask, cutmix_box2)

            mask_u_w_rvs = mask_u_w.clone()
            mask_u_w_rvs[valid_masks_pred_sq == 0] = ignore_index
            ignore_mask_rvs = ignore_mask.clone()
            ignore_mask_rvs[valid_masks_pred_sq == 0] = ignore_index

            validate_target_range(
                "loss_u_s1_target",
                mask_u_w_cutmixed1,
                cfg["nclass"],
                ignore_index,
                logger=logger,
                rank=rank,
            )
            validate_target_range(
                "loss_u_s2_target",
                mask_u_w_cutmixed2,
                cfg["nclass"],
                ignore_index,
                logger=logger,
                rank=rank,
            )
            validate_target_range(
                "loss_u_rvs_target",
                mask_u_w_rvs,
                cfg["nclass"],
                ignore_index,
                logger=logger,
                rank=rank,
            )

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = confidence_weighted_loss(
                loss_u_s1,
                conf_u_w_cutmixed1,
                ignore_mask_cutmixed1,
                ignore_index,
                conf_thresh=conf_thresh,
            )

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = confidence_weighted_loss(
                loss_u_s2,
                conf_u_w_cutmixed2,
                ignore_mask_cutmixed2,
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

            loss = (
                loss_x * 0.5
                + loss_u_s1 / 6.0
                + loss_u_s2 / 6.0
                + loss_u_rvs / 6.0
            )

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
                        "img_u_rvs": (img_u_rvs[0], Visualizer.TENSOR),
                        "mask_u_w_cutmixed1": (
                            mask_u_w_cutmixed1[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "mask_u_w_cutmixed2": (
                            mask_u_w_cutmixed2[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "mask_u_w_rvs": (
                            mask_u_w_rvs[0],
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
                        "pred_u_rvs": (
                            pred_u_rvs.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_recovered": (
                            pred_recovered.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s1.update(loss_u_s1.item())
            total_loss_s2.update(loss_u_s2.item())
            total_loss_rvs.update(loss_u_rvs.item())
            mask_ratio = (
                (conf_u_w >= conf_thresh) & (ignore_mask != ignore_index)
            ).sum().item() / (ignore_mask != ignore_index).sum().clamp(min=1).item()
            rvs_valid_ratio = valid_masks_pred_sq.mean().item()
            total_mask_ratio.update(mask_ratio)
            total_rvs_valid_ratio.update(rvs_valid_ratio)

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(
                    param_ema * ema_ratio + param.detach() * (1 - ema_ratio)
                )
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(
                    buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio)
                )

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_u_s1", loss_u_s1.item(), iters)
                writer.add_scalar("train/loss_u_s2", loss_u_s2.item(), iters)
                writer.add_scalar("train/loss_u_rvs", loss_u_rvs.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)
                writer.add_scalar("train/rvs_valid_ratio", rvs_valid_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, "
                    "Loss s1: {:.3f}, Loss s2: {:.3f}, Loss rvs: {:.3f}, "
                    "Mask ratio: {:.3f}, RVS valid: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s1.avg,
                        total_loss_s2.avg,
                        total_loss_rvs.avg,
                        total_mask_ratio.avg,
                        total_rvs_valid_ratio.avg,
                    )
                )

        mIoU, iou_class = validation_cpu(cfg, model, valloader)
        mIoU_ema, iou_class_ema = validation_cpu(cfg, model_ema, valloader)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, "
                    "EMA: {:.2f}".format(
                        cls_idx,
                        CLASSES[cfg["dataset"]][cls_idx],
                        iou,
                        iou_class_ema[cls_idx],
                    )
                )
            logger.info(
                "***** Evaluation ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n".format(
                    mIoU, mIoU_ema
                )
            )

            writer.add_scalar("eval/mIoU", mIoU, epoch)
            writer.add_scalar("eval/mIoU_ema", mIoU_ema, epoch)
            for cls_idx, iou in enumerate(iou_class):
                writer.add_scalar(
                    "eval/%s_IoU" % (CLASSES[cfg["dataset"]][cls_idx]), iou, epoch
                )
                writer.add_scalar(
                    "eval/%s_IoU_ema" % (CLASSES[cfg["dataset"]][cls_idx]),
                    iou_class_ema[cls_idx],
                    epoch,
                )

        is_best = mIoU >= previous_best

        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch

        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
                "previous_best_ema": previous_best_ema,
                "best_epoch": best_epoch,
                "best_epoch_ema": best_epoch_ema,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
