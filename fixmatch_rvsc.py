"""
FixMatch with RVS (Rotation-Variation-Scale) Augmentation
Converted to fixmatch.py style with DPT + DINOv2/v3 backbone and EMA Teacher.
"""

import argparse
from copy import deepcopy
import logging
import os
import pprint

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import torchvision.transforms.functional as TF

from dataset.semi_rvs import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.upernet import UperNet
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.focal import FocalLoss
from util.utils import count_params, init_log, AverageMeter, intersectionAndUnion
from util.dist_helper import setup_distributed
from util.train_utils import confidence_weighted_loss
from util.viz import Visualizer


def get_parser():
    parser = argparse.ArgumentParser(
        description="FixMatch with RVS Augmentation for Semi-Supervised Semantic Segmentation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def scale_back(pred_c_back, mask_c_back, size, box):
    """
    Scale back rotated/scaled predictions to original coordinate space.

    Args:
        pred_c_back: Predictions from context view [B, C, h, w]
        mask_c_back: Mask from context view [B, 1, h, w]
        size: Crop size
        box: Box parameters [x, y, x_c, y_c, s, theta]

    Returns:
        preds: Aligned predictions [B, C, H, W]
        masks: Valid masks [B, 1, H, W]
    """
    B, C, h, w = pred_c_back.shape

    preds = []
    masks = []

    for i in range(B):
        x, y, x_c, y_c, s, theta = box[i]
        x, y, x_c, y_c, s, theta = (
            x.item(),
            y.item(),
            x_c.item(),
            y_c.item(),
            s.item(),
            theta.item(),
        )

        pred = TF.rotate(
            pred_c_back[i], angle=-theta, interpolation=TF.InterpolationMode.BILINEAR
        )
        mask = TF.rotate(
            mask_c_back[i],
            angle=-theta,
            interpolation=TF.InterpolationMode.NEAREST,
            fill=0,
        )
        aligned_ctx = torch.zeros((C, h, w)).to(pred_c_back.device)
        aligned_mask = torch.zeros((1, h, w)).to(pred_c_back.device)
        rect_m = [x, y, x + size, y + size]
        rect_s = [x_c, y_c, x_c + size * s, y_c + size * s]

        inter_x1 = max(rect_m[0], rect_s[0])
        inter_y1 = max(rect_m[1], rect_s[1])
        inter_x2 = min(rect_m[2], rect_s[2])
        inter_y2 = min(rect_m[3], rect_s[3])

        m_x1 = int(round(inter_x1 - x))
        m_y1 = int(round(inter_y1 - y))
        m_x2 = int(round(inter_x2 - x))
        m_y2 = int(round(inter_y2 - y))

        target_h = m_y2 - m_y1
        target_w = m_x2 - m_x1

        c_x1 = int(round((inter_x1 - x_c) / s))
        c_y1 = int(round((inter_y1 - y_c) / s))
        c_x2 = int(round((inter_x2 - x_c) / s))
        c_y2 = int(round((inter_y2 - y_c) / s))

        c_x1, c_x2 = max(0, c_x1), min(size, c_x2)
        c_y1, c_y2 = max(0, c_y1), min(size, c_y2)

        pred_patch = pred[:, c_y1:c_y2, c_x1:c_x2]
        mask_patch = mask[:, c_y1:c_y2, c_x1:c_x2]

        if (
            target_h > 0
            and target_w > 0
            and pred_patch.shape[1] > 0
            and pred_patch.shape[2] > 0
        ):
            pred_patch = F.interpolate(
                pred_patch.unsqueeze(0), size=(target_h, target_w), mode="bilinear"
            ).squeeze(0)
            mask_patch = F.interpolate(
                mask_patch.unsqueeze(0), size=(target_h, target_w), mode="nearest"
            ).squeeze(0)

            aligned_ctx[:, m_y1:m_y2, m_x1:m_x2] = pred_patch
            aligned_mask[:, m_y1:m_y2, m_x1:m_x2] = mask_patch

        preds.append(aligned_ctx)
        masks.append(aligned_mask)

    return torch.stack(preds, dim=0), torch.stack(masks, dim=0)


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    """Validation function that matches fixmatch.py style."""
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()

    for x, y, _ in valid_loader:
        x = x.cuda()

        if cfg.get("eval_mode") == "slide_window":
            b, _, h, w = x.shape
            final = torch.zeros(b, cfg["nclass"], h, w).cuda()
            size = cfg["crop_size"]
            step = 510
            row = 0
            col = 0
            while row <= int(h / step):
                while col <= int(w / step):
                    sub_input = x[
                        :,
                        :,
                        min(row * step, h - size) : min(row * step + size, h),
                        min(col * step, w - size) : min(col * step + size, w),
                    ]
                    mask = model(sub_input)
                    final[
                        :,
                        :,
                        min(row * step, h - size) : min(row * step + size, h),
                        min(col * step, w - size) : min(col * step + size, w),
                    ] += mask
                    col += 1
                col = 0
                row += 1
            o = final.argmax(dim=1)
        elif cfg.get("eval_mode") == "resize":
            original_shape = x.shape[-2:]
            resized_x = F.interpolate(
                x, size=cfg["crop_size"], mode="bilinear", align_corners=True
            )
            resized_o = model(resized_x)
            o = F.interpolate(
                resized_o, size=original_shape, mode="bilinear", align_corners=True
            )
            o = o.argmax(dim=1)
        else:
            o = model(x)
            o = o.max(1)[1]

        gray = np.uint8(o.cpu().numpy())
        target = np.array(y, dtype=np.int32)
        intersection, union, target_area = intersectionAndUnion(
            gray, target, cfg["nclass"], cfg["ignore_index"]
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target_area)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)

    if cfg["dataset"] == "iSAID":
        mIoU = np.mean(iou_class[1:]) * 100.0
    else:
        mIoU = np.nanmean(iou_class) * 100.0

    return mIoU, iou_class * 100.0


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

    # Model configuration
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

    # EMA Teacher
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    # Loss functions
    ignore_index = cfg.get("ignore_index", 255)
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

    # Datasets
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
    conf_thresh = cfg["conf_thresh"]

    # Resume from checkpoint
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
        total_loss_x_rvs = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_s_rvs = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        model.train()

        for i, (
            (img_x, mask_x, img_x_c, mask_x_c),
            (img_u_w, img_u_s, img_u_c, ignore_mask, cutmix_box, _, box, mask_c),
        ) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_x_c, mask_x_c = img_x_c.cuda(), mask_x_c.cuda()
            img_u_w, img_u_s, img_u_c = img_u_w.cuda(), img_u_s.cuda(), img_u_c.cuda()
            ignore_mask = ignore_mask.cuda()
            mask_c = mask_c.cuda()

            # Get pseudo-labels from EMA teacher
            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            # Forward pass
            pred_x = model(img_x)
            pred_u_s = model(img_u_s)
            pred_x_rvs = model(img_x_c)
            pred_u_rvs = model(img_u_c)

            # Scale back RVS predictions
            pred_recovered, valid_masks = scale_back(
                pred_u_rvs, mask_c, cfg["crop_size"], box
            )

            # Compute confidence from recovered predictions (in original coordinate space)
            conf_u_rvs_recovered = pred_recovered.softmax(dim=1).max(dim=1)[
                0
            ]  # [B, H, W]

            valid_masks = valid_masks.squeeze(1)
            mask_u_w_rvs = mask_u_w.clone()
            mask_u_w_rvs[valid_masks == 0] = 255

            # Supervised losses
            loss_x = criterion_l(pred_x, mask_x)
            loss_x_rvs = criterion_l(pred_x_rvs, mask_x_c)

            # Unsupervised loss for strong augmentation
            loss_u_s = criterion_u(pred_u_s, mask_u_w)
            loss_u_s = confidence_weighted_loss(
                loss_u_s, conf_u_w, ignore_mask, ignore_index, conf_thresh=conf_thresh
            )

            loss_u_s_rvs = criterion_u(pred_recovered, mask_u_w_rvs)
            loss_u_s_rvs = confidence_weighted_loss(
                loss_u_s_rvs,
                conf_u_rvs_recovered,  # Use confidence from recovered predictions
                mask_u_w_rvs,
                ignore_index,
                conf_thresh=conf_thresh,
            )

            # Total loss
            # loss_x_rvs is detached to not participate in backward pass, but still logged
            loss = (loss_x + loss_x_rvs.detach() + loss_u_s + loss_u_s_rvs) / 4.0

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
                        "img_x_c": (img_x_c[0], Visualizer.TENSOR),
                        "mask_x_c": (mask_x_c[0], Visualizer.SEGMENTATION),
                        "pred_x_rvs": (
                            pred_x_rvs.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "img_u_w": (img_u_w[0], Visualizer.TENSOR),
                        "mask_u_w": (mask_u_w[0], Visualizer.SEGMENTATION),
                        "img_u_s": (img_u_s[0], Visualizer.TENSOR),
                        "pred_u_s": (
                            pred_u_s.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "img_u_c": (img_u_c[0], Visualizer.TENSOR),
                        "pred_u_rvs": (
                            pred_u_rvs.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_recovered": (
                            pred_recovered.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "mask_u_w_rvs": (mask_u_w_rvs[0], Visualizer.SEGMENTATION),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            # Update EMA teacher
            iters = epoch * len(trainloader_u) + i
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(
                    param_ema * ema_ratio + param.detach() * (1 - ema_ratio)
                )
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(
                    buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio)
                )

            # Update metrics
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_x_rvs.update(loss_x_rvs.item())
            total_loss_s.update(loss_u_s.item())
            total_loss_s_rvs.update(loss_u_s_rvs.item())

            mask_ratio = (
                (conf_u_w >= conf_thresh) & (ignore_mask != ignore_index)
            ).sum().item() / (ignore_mask != ignore_index).sum()
            total_mask_ratio.update(mask_ratio.item())

            # Learning rate schedule (polynomial decay)
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_x_rvs", loss_x_rvs.item(), iters)
                writer.add_scalar("train/loss_s", loss_u_s.item(), iters)
                writer.add_scalar("train/loss_s_rvs", loss_u_s_rvs.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, "
                    "Loss x_rvs: {:.3f}, Loss s: {:.3f}, Loss s_rvs: {:.3f}, "
                    "Mask ratio: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_x_rvs.avg,
                        total_loss_s.avg,
                        total_loss_s_rvs.avg,
                        total_mask_ratio.avg,
                    )
                )

        # Validation
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
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
