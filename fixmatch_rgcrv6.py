"""
FixMatch-RGCRv6: Geometry-Confidence Aware Feature Perturbation
for Semi-Supervised Semantic Segmentation

Combines:
- FixMatch / UniMatch-style EMA pseudo-labeling
- Dual strong-view consistency with independent CutMix targets
- RVS: Rotation-Variation-Scale geometric augmentation
- RGCR-native structured feature perturbation

Loss = loss_x + loss_u_s1 + loss_u_rvs
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
from model.semseg.my_upernet import MyUperNet
from model.semseg.upernet import UperNet
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
from util.validation import validation_cpu as shared_validation_cpu


RGCRV6_PERTURB = {
    "mask_strength": 0.6,
    "shift_strength": 0.15,
    "stable_floor": 0.35,
    "local_kernel": 3,
    "smooth_score": True,
    "s1_uncertainty_weight": 1.0,
    "s2_uncertainty_weight": 0.45,
    "s2_recovery_weight": 0.40,
    "s2_geometry_weight": 0.15,
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="FixMatch-RGCRv6: geometry-confidence aware RGCR "
        "for Semi-Supervised Semantic Segmentation"
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
    return shared_validation_cpu(cfg, model, valid_loader)


@torch.no_grad()
def project_weak_map_to_rvs(map_tensor, box, size, mode="nearest", fill_value=0):
    """Project weak-view maps into the rotated/scaled context view used by RVS."""
    device = map_tensor.device
    dtype = map_tensor.dtype
    is_spatial = map_tensor.dim() == 3
    if is_spatial:
        map_tensor = map_tensor.unsqueeze(1)

    bsz, channels, h, w = map_tensor.shape
    projected = torch.full(
        (bsz, channels, h, w), fill_value=fill_value, device=device, dtype=dtype
    )

    for i in range(bsz):
        x, y, x_c, y_c, s, theta = box[i]
        x, y, x_c, y_c, s, theta = (
            x.item(),
            y.item(),
            x_c.item(),
            y_c.item(),
            s.item(),
            theta.item(),
        )

        rect_m = [x, y, x + size, y + size]
        rect_s = [x_c, y_c, x_c + size * s, y_c + size * s]

        inter_x1 = max(rect_m[0], rect_s[0])
        inter_y1 = max(rect_m[1], rect_s[1])
        inter_x2 = min(rect_m[2], rect_s[2])
        inter_y2 = min(rect_m[3], rect_s[3])

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            rotated = TF.rotate(
                projected[i],
                angle=theta,
                interpolation=TF.InterpolationMode.NEAREST
                if mode == "nearest"
                else TF.InterpolationMode.BILINEAR,
                fill=fill_value,
            )
            projected[i] = rotated
            continue

        m_x1 = int(round(inter_x1 - x))
        m_y1 = int(round(inter_y1 - y))
        m_x2 = int(round(inter_x2 - x))
        m_y2 = int(round(inter_y2 - y))

        c_x1 = int(round((inter_x1 - x_c) / s))
        c_y1 = int(round((inter_y1 - y_c) / s))
        c_x2 = int(round((inter_x2 - x_c) / s))
        c_y2 = int(round((inter_y2 - y_c) / s))

        c_x1, c_x2 = max(0, c_x1), min(size, c_x2)
        c_y1, c_y2 = max(0, c_y1), min(size, c_y2)

        target_h = c_y2 - c_y1
        target_w = c_x2 - c_x1

        patch = map_tensor[i : i + 1, :, m_y1:m_y2, m_x1:m_x2]
        canvas = torch.full(
            (1, channels, h, w), fill_value=fill_value, device=device, dtype=dtype
        )

        if (
            target_h > 0
            and target_w > 0
            and patch.shape[-2] > 0
            and patch.shape[-1] > 0
        ):
            interp_kwargs = {"size": (target_h, target_w), "mode": mode}
            if mode in ("bilinear", "bicubic", "trilinear"):
                interp_kwargs["align_corners"] = False
            patch = F.interpolate(patch.float(), **interp_kwargs)
            if dtype.is_floating_point:
                patch = patch.to(dtype)
            else:
                patch = patch.round().to(dtype)
            canvas[:, :, c_y1:c_y2, c_x1:c_x2] = patch

        rotated = TF.rotate(
            canvas.squeeze(0),
            angle=theta,
            interpolation=TF.InterpolationMode.NEAREST
            if mode == "nearest"
            else TF.InterpolationMode.BILINEAR,
            fill=fill_value,
        )
        projected[i] = rotated

    if is_spatial:
        projected = projected.squeeze(1)
    return projected


def apply_cutmix_to_map(map_tensor, cutmix_box):
    mixed = map_tensor.clone()
    mixed[cutmix_box == 1] = map_tensor.flip(0)[cutmix_box == 1]
    return mixed


def binarize_valid_mask(mask_tensor, threshold=0.5):
    return (mask_tensor > threshold).to(mask_tensor.dtype)


def normalize_score_map(score_map, eps=1e-6):
    score_map = score_map.clamp(min=0.0)
    flat = score_map.flatten(1)
    max_val = flat.max(dim=1)[0].view(-1, 1, 1)
    return score_map / (max_val + eps)


def build_feature_perturbation(score_map):
    return {
        "score_map": normalize_score_map(score_map),
        "mask_strength": RGCRV6_PERTURB["mask_strength"],
        "shift_strength": RGCRV6_PERTURB["shift_strength"],
        "stable_floor": RGCRV6_PERTURB["stable_floor"],
        "local_kernel": RGCRV6_PERTURB["local_kernel"],
        "smooth_score": RGCRV6_PERTURB["smooth_score"],
    }


@torch.no_grad()
def validate_target_range(name, target, nclass, ignore_index, logger=None, rank=0):
    valid = (target != ignore_index) & ((target < 0) | (target >= nclass))
    if valid.any():
        bad_vals = torch.unique(target[valid]).detach().cpu().tolist()
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
    else:
        raise NotImplementedError(f"Unsupported model: {cfg['model']}")

    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth', map_location="cpu", weights_only=False)
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

    # EMA Teacher follows the same configurable model path as UniMatch v2
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

    criterion_u = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index).cuda(
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
        total_loss_u_s1 = AverageMeter()
        total_loss_u_rvs = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        model.train()

        for i, (
            (img_x, mask_x, _, _),
            (
                img_u_w,
                img_u_s1,
                img_u_rvs,
                ignore_mask,
                cutmix_box1,
                cutmix_box2,
                box,
                mask_c,
            ),
        ) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_rvs = (
                img_u_w.cuda(),
                img_u_s1.cuda(),
                img_u_rvs.cuda(),
            )
            ignore_mask = ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            box = box.cuda()
            mask_c = mask_c.cuda()

            iters = epoch * len(trainloader_u) + i

            # =====================
            # 1. EMA Teacher: Generate pseudo-labels from weak augmentation
            # =====================
            with torch.no_grad():
                pred_u_w_ema = model_ema(img_u_w).detach()

                conf_u_w = pred_u_w_ema.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w_ema.argmax(dim=1)

                mask_u_rvs_view = project_weak_map_to_rvs(
                    mask_u_w,
                    box,
                    cfg["crop_size"],
                    mode="nearest",
                    fill_value=ignore_index,
                )
                conf_u_rvs_view = project_weak_map_to_rvs(
                    conf_u_w, box, cfg["crop_size"], mode="bilinear", fill_value=0
                )
                ignore_mask_rvs_view = project_weak_map_to_rvs(
                    ignore_mask,
                    box,
                    cfg["crop_size"],
                    mode="nearest",
                    fill_value=ignore_index,
                )

            # =====================
            # 2. Student forward: labeled data
            # =====================
            pred_x = model(img_x)

            # =====================
            # 3. Student forward: dual strong augmentations with RGCR-native
            #    geometry-/confidence-aware feature perturbation
            # =====================
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = (
                img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            )

            mask_u_w_cutmixed1 = apply_cutmix_to_map(mask_u_w, cutmix_box1)
            conf_u_w_cutmixed1 = apply_cutmix_to_map(conf_u_w, cutmix_box1)
            ignore_mask_cutmixed1 = apply_cutmix_to_map(ignore_mask, cutmix_box1)

            with torch.no_grad():
                pred_u_rvs_seed = model_ema(img_u_rvs).detach()
                pred_seed_recovered, valid_masks_seed = scale_back(
                    pred_u_rvs_seed, mask_c, cfg["crop_size"], box
                )
                valid_masks_seed_sq = binarize_valid_mask(
                    valid_masks_seed.squeeze(1).float()
                )
                seed_recovered_conf = pred_seed_recovered.softmax(dim=1).max(dim=1)[0]
                seed_recovered_label = pred_seed_recovered.argmax(dim=1)
                seed_recovered_disagreement = (
                    (seed_recovered_label != mask_u_w).float() * valid_masks_seed_sq
                )
                seed_recovered_instability = (
                    0.5 * seed_recovered_disagreement
                    + 0.5 * (1.0 - seed_recovered_conf) * valid_masks_seed_sq
                )
                seed_recovered_instability_rvs = project_weak_map_to_rvs(
                    seed_recovered_instability,
                    box,
                    cfg["crop_size"],
                    mode="bilinear",
                    fill_value=0,
                )

            perturb_score_s1 = RGCRV6_PERTURB["s1_uncertainty_weight"] * (
                1.0 - conf_u_w_cutmixed1
            )
            perturb_score_rvs = (
                RGCRV6_PERTURB["s2_uncertainty_weight"]
                * (1.0 - conf_u_rvs_view)
                + RGCRV6_PERTURB["s2_recovery_weight"]
                * seed_recovered_instability_rvs
                + RGCRV6_PERTURB["s2_geometry_weight"]
                * (1.0 - mask_c.squeeze(1).float())
            )

            feature_perturb = build_feature_perturbation(
                torch.cat((perturb_score_s1, perturb_score_rvs), dim=0)
            )
            pred_u_s1, pred_u_rvs = model(
                torch.cat((img_u_s1, img_u_rvs)), feature_perturb=feature_perturb
            ).chunk(2)

            # =====================
            # 4. Student forward: RVS (rotated-varied-scaled) augmentation
            # =====================
            # pred_u_rvs is predicted from the pure RVS view (without CutMix)

            # =====================
            # 5. Scale-back: recover RVS predictions and features to original space
            # =====================
            pred_recovered, valid_masks_pred = scale_back(
                pred_u_rvs, mask_c, cfg["crop_size"], box
            )

            # =====================
            # 6. Prepare masks for RVS pseudo-label loss
            # =====================
            valid_masks_pred_sq = binarize_valid_mask(
                valid_masks_pred.squeeze(1).float()
            )  # [B, H, W]

            # mask_u_w_rvs: pseudo-labels for RVS branch
            # In valid regions: use EMA teacher pseudo-labels (mask_u_w)
            # In invalid regions: set to ignore_index
            mask_u_w_rvs = mask_u_w.clone()
            mask_u_w_rvs[valid_masks_pred_sq == 0] = ignore_index

            # Combine dataset ignore regions with RVS invalid regions
            ignore_mask_rvs = ignore_mask.clone()
            ignore_mask_rvs[valid_masks_pred_sq == 0] = ignore_index

            # =====================
            # 7. Compute losses
            # =====================

            # --- Supervised loss: loss_x ---
            loss_x = criterion_l(pred_x, mask_x)

            # --- Unsupervised loss (strong view 1, same crop space) ---
            validate_target_range(
                "loss_u_s1_target",
                mask_u_w_cutmixed1,
                cfg["nclass"],
                ignore_index,
                logger=logger,
                rank=rank,
            )

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = confidence_weighted_loss(
                loss_u_s1,
                conf_u_w_cutmixed1,
                ignore_mask_cutmixed1,
                ignore_index,
                conf_thresh=conf_thresh,
            )

            # strong2 already serves as the RVS-space branch; do not add a second
            # pseudo-label loss on the same prediction path to avoid duplicate supervision.

            # --- Unsupervised loss (RVS): loss_u_rvs ---
            validate_target_range(
                "loss_u_rvs_target",
                mask_u_w_rvs,
                cfg["nclass"],
                ignore_index,
                logger=logger,
                rank=rank,
            )
            loss_u_rvs = criterion_u(pred_recovered, mask_u_w_rvs)
            loss_u_rvs = confidence_weighted_loss(
                loss_u_rvs,
                conf_u_w,
                ignore_mask_rvs,
                ignore_index,
                conf_thresh=conf_thresh,
            )

            # =====================
            # 8. Total loss
            # =====================
            loss = (
                loss_x * 0.5
                + loss_u_s1 * 0.25
                + loss_u_rvs * 0.25
            )

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
                        # RVS / strong2: transformed context image and predictions
                        "img_u_rvs": (img_u_rvs[0], Visualizer.TENSOR),
                        "pred_u_rvs": (
                            pred_u_rvs.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_recovered": (
                            pred_recovered.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        # Unlabeled: weak image, pseudo-label, strong predictions
                        "img_u_w": (img_u_w[0], Visualizer.TENSOR),
                        "mask_u_w": (mask_u_w[0], Visualizer.SEGMENTATION),
                        "img_u_s1": (img_u_s1[0], Visualizer.TENSOR),
                        "pred_u_s1": (
                            pred_u_s1.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "mask_u_w_cutmixed1": (
                            mask_u_w_cutmixed1[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "perturb_score_s1": (
                            perturb_score_s1[0].unsqueeze(0),
                            Visualizer.TENSOR,
                        ),
                        "perturb_score_rvs": (
                            perturb_score_rvs[0].unsqueeze(0),
                            Visualizer.TENSOR,
                        ),
                        # RVS pseudo-label target (ignore_index in invalid regions)
                        "mask_u_w_rvs": (mask_u_w_rvs[0], Visualizer.SEGMENTATION),
                        "valid_mask_rvs": (
                            valid_masks_pred_sq[0].unsqueeze(0).float(),
                            Visualizer.TENSOR,
                        ),
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
                    seed_disagree_ratio = (
                        seed_recovered_disagreement[valid_masks_seed_sq > 0]
                        .float()
                        .mean()
                        .item()
                        if (valid_masks_seed_sq > 0).any()
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
                        f"seed_disagree={seed_disagree_ratio:.3f}, "
                        f"perturb_s1={perturb_score_s1.mean().item():.3f}, "
                        f"perturb_rvs={perturb_score_rvs.mean().item():.3f}, "
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
            total_loss_u_s1.update(loss_u_s1.item())
            total_loss_u_rvs.update(loss_u_rvs.item())

            valid_s1 = (ignore_mask != ignore_index).sum().item()
            valid_s2 = (ignore_mask_rvs_view != ignore_index).sum().item()
            mask_ratio_s1 = (
                ((conf_u_w >= conf_thresh) & (ignore_mask != ignore_index)).sum().item()
                / max(valid_s1, 1)
            )
            mask_ratio_s2 = (
                (
                    (conf_u_rvs_view >= conf_thresh)
                    & (ignore_mask_rvs_view != ignore_index)
                ).sum().item()
                / max(valid_s2, 1)
            )
            mask_ratio = (mask_ratio_s1 + mask_ratio_s2) / 2.0
            total_mask_ratio.update(mask_ratio)

            # Learning rate schedule (polynomial decay)
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_u_s1", loss_u_s1.item(), iters)
                writer.add_scalar("train/loss_u_rvs", loss_u_rvs.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, "
                    "Loss u_s1: {:.3f}, Loss u_rvs: {:.3f}, "
                    "Mask ratio: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_u_s1.avg,
                        total_loss_u_rvs.avg,
                        total_mask_ratio.avg,
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
