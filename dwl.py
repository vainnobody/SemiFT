import argparse
from copy import deepcopy
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi_rs import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.upernet import UperNet
from supervised import evaluate, validation_cpu
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.focal import FocalLoss
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.dwl_utils import (
    init_cls_memory,
    update_cls_memory,
    sample_cls_bins,
    calc_wgt_bins,
    downsample_for_memory,
)

from util.viz import Visualizer


# ========================= DWL Hardcoded Parameters =========================
MEMORY_N_BATCHES = 50  # Number of batches to keep in class memory
MEMORY_DOWNSAMPLE_SIZE = 64  # Spatial size for downsampling before memory update
ALPHA_HEAD_TRANSFER = 0.5  # Alpha for head parameter transfer (0 = no transfer)
# =============================================================================


def get_parser():
    parser = argparse.ArgumentParser(
        description="DWL (Distribution-aware Weighting) for Semi-Supervised Semantic Segmentation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


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
    # DINOv2 uses patch_size=14, DINOv3 uses patch_size=16
    patch_size = 14 if backbone_version == "dinov2" else 16

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

    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)

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
    valset = ValDataset(
        cfg["dataset"], cfg["data_root"], "val", ignore_value=cfg["ignore_index"]
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

    # Initialize DWL class-wise memory banks
    cls_memory_u = init_cls_memory(
        cfg["nclass"], device=torch.device("cuda", local_rank)
    )

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        model_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        previous_best_ema = checkpoint["previous_best_ema"]
        best_epoch = checkpoint["best_epoch"]
        best_epoch_ema = checkpoint["best_epoch_ema"]
        # Restore class memory if saved
        if "cls_memory_u" in checkpoint:
            cls_memory_u = checkpoint["cls_memory_u"]

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
        total_loss_s = AverageMeter()
        total_wgt_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        model.train()

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _),
        ) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()

            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                prob_u_w = pred_u_w.softmax(dim=1)
                conf_u_w = prob_u_w.max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            # ===================== DWL: Update class memory and compute weights =====================
            iters = epoch * len(trainloader_u) + i

            # Downsample for memory efficiency
            prob_uw_bar, pl_uw_bar = downsample_for_memory(
                prob_u_w, mask_u_w, target_size=MEMORY_DOWNSAMPLE_SIZE
            )

            # Update class memory with pseudo-label predictions
            cls_memory_u = update_cls_memory(
                cls_memory_u, prob_uw_bar.detach(), pl_uw_bar, MEMORY_N_BATCHES
            )

            # Sample distribution bins and compute weights
            cls_bins_u = sample_cls_bins(cls_memory_u)

            # Flatten confidence and pseudo-labels for weight computation
            conf_u_w_flat = (
                F.interpolate(
                    conf_u_w.unsqueeze(1),
                    size=(MEMORY_DOWNSAMPLE_SIZE, MEMORY_DOWNSAMPLE_SIZE),
                    mode="nearest",
                )
                .squeeze(1)
                .reshape(-1)
            )
            pl_uw_flat = pl_uw_bar

            # Compute distribution-aware weights
            wgt_u_flat = calc_wgt_bins(
                cls_bins_u, conf_u_w_flat, pl_uw_flat, iters, total_iters
            )

            # Reshape weights back to spatial dimensions
            b = img_u_w.shape[0]
            wgt_u_spatial = wgt_u_flat.reshape(
                b, MEMORY_DOWNSAMPLE_SIZE, MEMORY_DOWNSAMPLE_SIZE
            )
            wgt_u = F.interpolate(
                wgt_u_spatial.unsqueeze(1),
                size=(img_u_w.shape[2], img_u_w.shape[3]),
                mode="nearest",
            ).squeeze(1)

            # Apply warmup: no unsupervised loss before memory is populated
            if iters <= MEMORY_N_BATCHES:
                wgt_u = wgt_u * 0.0
            # ==========================================================================================

            # CutMix augmentation
            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = img_u_s.flip(
                0
            )[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

            num_lb, num_ulb = img_x.shape[0], img_u_s.shape[0]
            pred_x, pred_u_s = model(torch.cat((img_x, img_u_s))).split(
                [num_lb, num_ulb]
            )

            mask_u_w_cutmixed, wgt_u_cutmixed, ignore_mask_cutmixed = (
                mask_u_w.clone(),
                wgt_u.clone(),
                ignore_mask.clone(),
            )

            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w.flip(0)[cutmix_box == 1]
            wgt_u_cutmixed[cutmix_box == 1] = wgt_u.flip(0)[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask.flip(0)[cutmix_box == 1]

            loss_x = criterion_l(pred_x, mask_x)

            # DWL: Use distribution-aware weights instead of confidence threshold
            loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed)
            loss_mask = ignore_mask_cutmixed != 255
            # Clamp weights to [0, 1] for loss weighting (sigmoid output is in [-1, 1])
            wgt_u_positive = (wgt_u_cutmixed + 1) / 2  # Map from [-1, 1] to [0, 1]
            loss_u_s = (
                loss_u_s * wgt_u_positive * loss_mask
            ).sum() / loss_mask.sum().clamp(min=1.0)

            loss = (loss_x + loss_u_s) / 2.0

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
                        "img_u_s": (img_u_s[0], Visualizer.TENSOR),
                        "mask_u_w_cutmixed": (
                            mask_u_w_cutmixed[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_u_s": (
                            pred_u_s.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            # Track average weight instead of mask ratio
            wgt_avg = (
                wgt_u_positive[loss_mask].mean().item() if loss_mask.sum() > 0 else 0.0
            )
            total_wgt_ratio.update(wgt_avg)

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
                writer.add_scalar("train/loss_s", loss_u_s.item(), iters)
                writer.add_scalar("train/wgt_avg", wgt_avg, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Wgt avg: "
                    "{:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_wgt_ratio.avg,
                    )
                )

        # ===================== DWL: Alpha-based head transfer (optional) =====================
        if ALPHA_HEAD_TRANSFER > 0:
            # Transfer EMA head parameters to main model head with alpha weighting
            # This mimics RS-DWL's pseudo_classifier transfer mechanism
            for param, param_ema in zip(
                model.module.head.parameters(), model_ema.module.head.parameters()
            ):
                param.data = param_ema.data * ALPHA_HEAD_TRANSFER + param.data * (
                    1 - ALPHA_HEAD_TRANSFER
                )
        # ======================================================================================

        eval_mode = "sliding_window" if cfg["dataset"] == "cityscapes" else "original"

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
                "***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n".format(
                    eval_mode, mIoU, mIoU_ema
                )
            )

            writer.add_scalar("eval/mIoU", mIoU, epoch)
            writer.add_scalar("eval/mIoU_ema", mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar(
                    "eval/%s_IoU" % (CLASSES[cfg["dataset"]][i]), iou, epoch
                )
                writer.add_scalar(
                    "eval/%s_IoU_ema" % (CLASSES[cfg["dataset"]][i]),
                    iou_class_ema[i],
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
                "cls_memory_u": cls_memory_u,  # Save class memory for resume
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
