"""
FixMatch-RGCRv3: Rotation-Geometric Consistency Regularization
with Geo-Topo Critical Reweighting for Semi-Supervised Semantic Segmentation

Combines:
- FixMatch: Pseudo-label based semi-supervised learning
- RVS: Rotation-Variation-Scale geometric augmentation
- Feature Perturbation (FP): Dropout-based feature diversity
- GTCR: Geo-Topo Critical Reweighting over RGCR-stable pixels

Loss = RGCR base loss with topology-aware reweighting on
loss_u_s, loss_u_rvs, and loss_fp, while keeping the original
RGCR loss structure unchanged.
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

from dataset.semi_rvs import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt_rankmatch import DPT_RankMatch
from model.semseg.rgcr_utils import (
    scale_back,
)  # , scale_back_features, geometric_corr_loss

from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.focal import FocalLoss
from util.utils import count_params, init_log, AverageMeter, intersectionAndUnion
from util.dist_helper import setup_distributed
from util.train_utils import confidence_weighted_loss
from util.viz import Visualizer


def get_parser():
    parser = argparse.ArgumentParser(
        description="FixMatch-RGCRv3: Rotation-Geometric Consistency Regularization "
        "with Geo-Topo Critical Reweighting"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()

    for x, y, _ in valid_loader:
        x = x.cuda()
        if cfg["eval_mode"] == "slide_window":
            b, _, h, w = x.shape  # 获取输入图像的尺寸 (batch, channels, height, width)
            final = torch.zeros(b, cfg["nclass"], h, w).cuda()  # 用于存储最终预测结果
            size = cfg["crop_size"]
            step = 510
            b = 0
            a = 0
            while a <= int(h / step):
                while b <= int(w / step):
                    sub_input = x[
                        :,
                        :,
                        min(a * step, h - size) : min(a * step + size, h),
                        min(b * step, w - size) : min(b * step + size, w),
                    ]
                    # print("sub_input.shape", sub_input.shape)
                    mask = model(sub_input)[0]
                    final[
                        :,
                        :,
                        min(a * step, h - size) : min(a * step + size, h),
                        min(b * step, w - size) : min(b * step + size, w),
                    ] += mask
                    b += 1
                b = 0
                a += 1
            o = final.argmax(dim=1)

        elif cfg["eval_mode"] == "resize":
            # 使用缩放方式进行预测
            original_shape = x.shape[-2:]  # 保存原始图像的尺寸 (h, w)
            resized_x = F.interpolate(
                x, size=cfg["crop_size"], mode="bilinear", align_corners=True
            )
            resized_o = model(resized_x)[0]
            # 将预测结果复原到原始尺寸
            o = F.interpolate(
                resized_o, size=original_shape, mode="bilinear", align_corners=True
            )
            o = o.argmax(dim=1)

        else:
            # 直接进行预测（非滑动窗口模式）

            o = model(x)[0]
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

    return mIoU, iou_class


def _shift_mask(mask, direction, fill_value):
    if direction == "up":
        return F.pad(mask[:, 1:, :], (0, 0, 0, 1), value=fill_value)
    if direction == "down":
        return F.pad(mask[:, :-1, :], (0, 0, 1, 0), value=fill_value)
    if direction == "left":
        return F.pad(mask[:, :, 1:], (0, 1, 0, 0), value=fill_value)
    if direction == "right":
        return F.pad(mask[:, :, :-1], (1, 0, 0, 0), value=fill_value)
    raise ValueError(f"Unknown direction: {direction}")


def build_boundary_map(mask, ignore_index=255):
    valid = mask != ignore_index
    boundary = torch.zeros_like(valid)
    for direction in ["up", "down", "left", "right"]:
        neigh = _shift_mask(mask, direction, ignore_index)
        boundary |= valid & (neigh != ignore_index) & (neigh != mask)
    return boundary.float()


def build_thin_structure_map(mask, ignore_index=255, max_same_neighbors=1):
    valid = mask != ignore_index
    same_neighbors = torch.zeros_like(mask, dtype=torch.float32)
    for direction in ["up", "down", "left", "right"]:
        neigh = _shift_mask(mask, direction, ignore_index)
        same_neighbors += (valid & (neigh == mask) & (neigh != ignore_index)).float()
    thin = valid & (same_neighbors <= float(max_same_neighbors))
    return thin.float()


def build_topo_context_map(core_map, kernel_size=5):
    if kernel_size <= 1:
        return torch.zeros_like(core_map)
    pad = kernel_size // 2
    dilated = F.max_pool2d(core_map.unsqueeze(1), kernel_size, stride=1, padding=pad)
    context = (dilated.squeeze(1) > 0).float() - (core_map > 0).float()
    return context.clamp(min=0.0)


def build_geo_topo_weight(
    mask_u_w,
    conf_u_w,
    pred_recovered,
    valid_masks_pred_sq,
    ignore_mask,
    ignore_index,
    conf_thresh,
    boundary_weight=1.0,
    thin_weight=1.0,
    context_weight=0.5,
    context_kernel=5,
    conf_floor=0.6,
    thin_neighbors=1,
    max_scale=3.0,
):
    pred_recovered_label = pred_recovered.argmax(dim=1)
    geo_agree = (
        (valid_masks_pred_sq > 0)
        & (ignore_mask != ignore_index)
        & (pred_recovered_label == mask_u_w)
        & (conf_u_w >= conf_floor)
    )

    boundary_map = build_boundary_map(mask_u_w, ignore_index=ignore_index)
    thin_map = build_thin_structure_map(
        mask_u_w, ignore_index=ignore_index, max_same_neighbors=thin_neighbors
    )
    topo_core = ((boundary_map > 0) | (thin_map > 0)).float()
    context_map = build_topo_context_map(topo_core, kernel_size=context_kernel)

    topo_weight = torch.ones_like(conf_u_w)
    topo_bonus = boundary_weight * boundary_map + thin_weight * thin_map + context_weight * context_map
    topo_weight = topo_weight + topo_bonus * geo_agree.float()

    conf_gate = ((conf_u_w >= conf_thresh).float() + 0.25 * ((conf_u_w >= conf_floor) & (conf_u_w < conf_thresh)).float())
    topo_weight = 1.0 + (topo_weight - 1.0) * conf_gate
    topo_weight = topo_weight.clamp(min=1.0, max=max_scale)
    return topo_weight, boundary_map, thin_map, context_map, geo_agree


def weighted_confidence_loss(
    loss_map,
    conf_map,
    ignore_mask,
    topo_weight,
    ignore_index,
    conf_thresh=0.95,
):
    valid = (conf_map >= conf_thresh) & (ignore_mask != ignore_index)
    return (loss_map * topo_weight * valid.float()).sum() / valid.sum().clamp(min=1.0)


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

    # Use DPT_RankMatch which supports need_fp and returns features
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

    # EMA Teacher (uses standard DPT_RankMatch, returns pred + feat)
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

    # RGCR hyperparameters (hardcoded)
    # geo_corr_weight = 0.1  # Weight for geometric correlation loss
    # num_landmarks = 64  # Number of orthogonal landmarks
    # rank_k = 4  # Top-k for rank computation

    # Datasets (use semi_rvs for geometric augmentation)
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
    gtcr_enable = cfg.get("gtcr_enable", True)
    gtcr_boundary_weight = cfg.get("gtcr_boundary_weight", 1.0)
    gtcr_thin_weight = cfg.get("gtcr_thin_weight", 1.0)
    gtcr_context_weight = cfg.get("gtcr_context_weight", 0.5)
    gtcr_context_kernel = cfg.get("gtcr_context_kernel", 5)
    gtcr_thin_neighbors = cfg.get("gtcr_thin_neighbors", 1)
    gtcr_conf_floor = cfg.get("gtcr_conf_floor", 0.6)
    gtcr_max_scale = cfg.get("gtcr_max_scale", 3.0)

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
        total_loss_u_s = AverageMeter()
        total_loss_u_rvs = AverageMeter()
        total_loss_fp = AverageMeter()
        # total_loss_geo_corr = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_gtcr_weight = AverageMeter()
        total_gtcr_boundary = AverageMeter()
        total_gtcr_thin = AverageMeter()
        total_gtcr_geo = AverageMeter()

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
            cutmix_box = cutmix_box.cuda()
            mask_c = mask_c.cuda()

            iters = epoch * len(trainloader_u) + i

            # =====================
            # 1. EMA Teacher: Generate pseudo-labels from weak augmentation
            # =====================
            with torch.no_grad():
                # model_ema returns (pred, feat) via DPT_RankMatch.forward
                pred_u_w_ema, feat_u_w_ema = model_ema(img_u_w)
                pred_u_w_ema = pred_u_w_ema.detach()
                feat_u_w_ema = feat_u_w_ema.detach()

                # Pseudo-labels from EMA teacher
                # mask_u_w: [B, H, W] - argmax class indices as pseudo-labels
                # conf_u_w: [B, H, W] - max softmax probability as confidence
                conf_u_w = pred_u_w_ema.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w_ema.argmax(dim=1)

            # =====================
            # 2. Student forward: labeled + unlabeled weak (with FP)
            # =====================
            # need_fp=True returns (pred, pred_fp, feat)
            # Concatenate labeled and unlabeled weak images
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp, feats = model(torch.cat((img_x, img_u_w)), need_fp=True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            _, feat_u_w = feats.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]  # FP prediction for unlabeled only

            # =====================
            # 3. Student forward: strong augmentation (with CutMix)
            # =====================
            # Apply CutMix to strong augmentation image
            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = img_u_s.flip(
                0
            )[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

            pred_u_s = model(img_u_s)[0]  # returns (pred, feat), take pred

            # =====================
            # 4. Student forward: RVS (rotated-varied-scaled) augmentation
            # =====================
            pred_u_rvs, feat_u_rvs = model(img_u_c)  # returns (pred, feat)

            # =====================
            # 5. Scale-back: recover RVS predictions and features to original space
            # =====================
            # Recover predictions for pseudo-label loss
            pred_recovered, valid_masks_pred = scale_back(
                pred_u_rvs, mask_c, cfg["crop_size"], box
            )

            # # Recover features for geometric correlation loss
            # # NOTE: feat_u_rvs has backbone spatial resolution (crop_size/patch_size),
            # # but scale_back uses pixel-space coordinates from box.
            # # Must interpolate to full resolution first.
            # feat_u_rvs_full = F.interpolate(
            #     feat_u_rvs,
            #     size=pred_u_rvs.shape[-2:],
            #     mode="bilinear",
            #     align_corners=True,
            # )
            # feat_recovered, _ = scale_back_features(
            #     feat_u_rvs_full, mask_c, cfg["crop_size"], box
            # )

            # =====================
            # 6. Prepare masks for RVS pseudo-label loss
            # =====================
            valid_masks_pred_sq = valid_masks_pred.squeeze(1)  # [B, H, W]

            # mask_u_w_rvs: pseudo-labels for RVS branch
            # In valid regions: use EMA teacher pseudo-labels (mask_u_w)
            # In invalid regions: set to ignore_index (255)
            mask_u_w_rvs = mask_u_w.clone()
            mask_u_w_rvs[valid_masks_pred_sq == 0] = 255

            # Combine dataset ignore regions with RVS invalid regions
            ignore_mask_rvs = ignore_mask.clone()
            ignore_mask_rvs[valid_masks_pred_sq == 0] = ignore_index

            # =====================
            # 6.5 Geo-Topo Critical Reweighting (GTCR)
            # =====================
            if gtcr_enable:
                topo_weight, boundary_map, thin_map, context_map, geo_agree_mask = build_geo_topo_weight(
                    mask_u_w,
                    conf_u_w,
                    pred_recovered.detach(),
                    valid_masks_pred_sq,
                    ignore_mask,
                    ignore_index,
                    conf_thresh,
                    boundary_weight=gtcr_boundary_weight,
                    thin_weight=gtcr_thin_weight,
                    context_weight=gtcr_context_weight,
                    context_kernel=gtcr_context_kernel,
                    conf_floor=gtcr_conf_floor,
                    thin_neighbors=gtcr_thin_neighbors,
                    max_scale=gtcr_max_scale,
                )
            else:
                topo_weight = torch.ones_like(conf_u_w)
                boundary_map = torch.zeros_like(conf_u_w)
                thin_map = torch.zeros_like(conf_u_w)
                context_map = torch.zeros_like(conf_u_w)
                geo_agree_mask = (valid_masks_pred_sq > 0) & (ignore_mask != ignore_index)

            # =====================
            # 7. Compute losses
            # =====================

            # --- Supervised loss: loss_x ---
            loss_x = criterion_l(pred_x, mask_x)

            # --- Unsupervised loss (strong aug with CutMix): loss_u_s ---
            # CutMix pseudo-labels, confidence, and ignore_mask for strong augmentation
            mask_u_w_cutmixed = mask_u_w.clone()
            conf_u_w_cutmixed = conf_u_w.clone()
            ignore_mask_cutmixed = ignore_mask.clone()

            topo_weight_cutmixed = topo_weight.clone()

            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w.flip(0)[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w.flip(0)[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask.flip(0)[cutmix_box == 1]
            topo_weight_cutmixed[cutmix_box == 1] = topo_weight.flip(0)[cutmix_box == 1]

            loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed)
            loss_u_s = weighted_confidence_loss(
                loss_u_s,
                conf_u_w_cutmixed,
                ignore_mask_cutmixed,
                topo_weight_cutmixed,
                ignore_index,
                conf_thresh=conf_thresh,
            )

            # --- Unsupervised loss (RVS): loss_u_rvs ---
            # Fix: use EMA teacher confidence (conf_u_w) instead of student confidence
            # Fix: use ignore_mask_rvs (dataset ignore + RVS invalid) instead of mask_u_w_rvs
            loss_u_rvs = criterion_u(pred_recovered, mask_u_w_rvs)
            loss_u_rvs = weighted_confidence_loss(
                loss_u_rvs,
                conf_u_w,
                ignore_mask_rvs,
                topo_weight,
                ignore_index,
                conf_thresh=conf_thresh,
            )

            # --- Feature perturbation loss: loss_fp ---
            # pred_u_w_fp: [B, nclass, H, W] FP prediction for unlabeled weak
            # mask_u_w: [B, H, W] pseudo-labels from EMA teacher
            loss_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_fp = weighted_confidence_loss(
                loss_fp,
                conf_u_w,
                ignore_mask,
                topo_weight,
                ignore_index,
                conf_thresh=conf_thresh,
            )

            # # --- Geometric correlation loss: loss_geo_corr ---
            # # Resize features to same spatial resolution for correlation computation
            # H, W = pred_u_w.shape[-2:]
            # # Detach feat_u_w: it's the reference (like RankMatch detaches weak features)
            # # Gradients only flow through feat_recovered (RVS student branch)
            # feat_u_w_resized = F.interpolate(
            #     feat_u_w.detach(), size=(H, W), mode="bilinear", align_corners=True
            # )
            # feat_recovered_resized = F.interpolate(
            #     feat_recovered, size=(H, W), mode="bilinear", align_corners=True
            # )

            # loss_geo_corr = geometric_corr_loss(
            #     feat_u_w_resized,
            #     feat_recovered_resized,
            #     local_rank,
            #     num_landmarks,
            #     rank_k,
            # )

            # =====================
            # 8. Total loss (following RankMatch-style weighting)
            # =====================
            # loss_x + (loss_u_s + loss_u_rvs) + loss_fp
            # Following rankmatch weighting style:
            #   supervised:    loss_x
            #   pseudo-label:  (loss_u_s * 0.25 + loss_u_rvs * 0.25)
            #   FP:            loss_fp * 0.5
            loss = (loss_x + loss_u_s * 0.25 + loss_u_rvs * 0.25 + loss_fp * 0.5) / 2.0
            # + geo_corr_weight * loss_geo_corr  # disabled to reduce memory

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # =====================
            # 8.5 Visualization (first 10 iters per epoch)
            # =====================
            if i < 10:
                viz.push(
                    {
                        # Supervised: labeled image, GT, prediction
                        "img_x": (img_x[0], Visualizer.TENSOR),
                        "mask_x": (mask_x[0], Visualizer.SEGMENTATION),
                        "pred_x": (pred_x.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                        # RVS: context image, RVS prediction, recovered prediction
                        "img_u_c": (img_u_c[0], Visualizer.TENSOR),
                        "pred_u_rvs": (
                            pred_u_rvs.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_recovered": (
                            pred_recovered.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        # Unlabeled: weak image, pseudo-label, strong pred, FP pred
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
                        # RVS pseudo-label target (255 in invalid regions)
                        "mask_u_w_rvs": (mask_u_w_rvs[0], Visualizer.SEGMENTATION),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            # RVS diagnostic logging
            if rank == 0 and i % 100 == 0:
                with torch.no_grad():
                    valid_ratio = valid_masks_pred_sq.float().mean().item()
                    conf_in_valid = conf_u_w[valid_masks_pred_sq > 0]
                    high_conf_ratio = (
                        (conf_in_valid >= conf_thresh).float().mean().item()
                        if conf_in_valid.numel() > 0
                        else 0
                    )
                    agree_ratio = (
                        (pred_recovered.argmax(1) == mask_u_w)[valid_masks_pred_sq > 0]
                        .float()
                        .mean()
                        .item()
                        if (valid_masks_pred_sq > 0).any()
                        else 0
                    )
                    logger.info(
                        f"[RVS Debug] valid_ratio={valid_ratio:.3f}, "
                        f"high_conf_valid={high_conf_ratio:.3f}, "
                        f"student-teacher agree={agree_ratio:.3f}"
                    )

            # =====================
            # 9. Update EMA teacher
            # =====================
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(
                    param_ema * ema_ratio + param.detach() * (1 - ema_ratio)
                )
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(
                    buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio)
                )

            # =====================
            # 10. Logging
            # =====================
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_u_s.update(loss_u_s.item())
            total_loss_u_rvs.update(loss_u_rvs.item())
            total_loss_fp.update(loss_fp.item())
            # total_loss_geo_corr.update(loss_geo_corr.item())
            total_gtcr_weight.update(topo_weight.mean().item())
            total_gtcr_boundary.update(boundary_map.mean().item())
            total_gtcr_thin.update(thin_map.mean().item())
            total_gtcr_geo.update(geo_agree_mask.float().mean().item())

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
                writer.add_scalar("train/loss_u_s", loss_u_s.item(), iters)
                writer.add_scalar("train/loss_u_rvs", loss_u_rvs.item(), iters)
                writer.add_scalar("train/loss_fp", loss_fp.item(), iters)
                # writer.add_scalar("train/loss_geo_corr", loss_geo_corr.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)
                writer.add_scalar("train/gtcr_weight_mean", topo_weight.mean().item(), iters)
                writer.add_scalar("train/gtcr_boundary_ratio", boundary_map.mean().item(), iters)
                writer.add_scalar("train/gtcr_thin_ratio", thin_map.mean().item(), iters)
                writer.add_scalar("train/gtcr_geo_agree_ratio", geo_agree_mask.float().mean().item(), iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, "
                    "Loss u_s: {:.3f}, Loss u_rvs: {:.3f}, Loss fp: {:.3f}, "
                    "Mask ratio: {:.3f}, GTCR_w: {:.3f}, GTCR_b: {:.3f}, GTCR_t: {:.3f}, GTCR_g: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_u_s.avg,
                        total_loss_u_rvs.avg,
                        total_loss_fp.avg,
                        total_mask_ratio.avg,
                        total_gtcr_weight.avg,
                        total_gtcr_boundary.avg,
                        total_gtcr_thin.avg,
                        total_gtcr_geo.avg,
                    )
                )

        # =====================
        # Validation
        # =====================
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
