"""
SegMind: Semi-supervised Semantic Segmentation with Entropy-guided Learning.
Converted to FixMatch style for SemiFT project.

This script implements SegMind's multi-loss training strategy:
- loss_l: Supervised CrossEntropy loss
- loss_e: Entropy distillation MSE loss
- loss_r: Masked reconstruction MSE loss
- loss_rsc: Reconstruction prediction CrossEntropy loss
- loss_c: Contrastive learning loss with memory bank
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

from dataset.semi_rs import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt_segmind import DPT_SegMind
from supervised import evaluate, validation_cpu
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.focal import FocalLoss
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.segmind_utils import (
    get_batch_mask_tensor,
    generate_u_data,
    cal_c_loss,
    init_memory_bank,
)


# ============================================================================
# SegMind Hardcoded Parameters
# ============================================================================
SEGMIND_CONFIG = {
    "conf_thresh": 0.95,  # Pseudo label confidence threshold
    "lambda_l": 1.0,  # Supervised loss weight
    "lambda_e": 1.0,  # Entropy loss weight (0 = disable)
    "lambda_r": 1.0,  # Reconstruction loss weight (0 = disable)
    "lambda_rsc": 1.0,  # Reconstruction segmentation loss weight
    "lambda_c": 1.0,  # Contrastive loss weight (0 = disable)
    "mask_rate": 0.75,  # Mask ratio for reconstruction
    "mask_gap": 16,  # Mask block size
    "epoch_pre": 50,  # Epochs to use reconstruction loss
    "query_threshold": 0.97,  # Contrastive learning query threshold
    "temperature": 0.5,  # Contrastive loss temperature
    "bank_size": 10000,  # Memory bank size per class
    "num_query": 256,  # Number of query samples
    "num_negative": 512,  # Number of negative samples
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="SegMind: Semi-supervised Semantic Segmentation with Entropy-guided Learning"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def get_pseudo_labels(model_ema, img_l_w, img_u_w, n_l, hw):
    """
    Get pseudo labels from EMA teacher model.

    Args:
        model_ema: EMA teacher model
        img_l_w: Labeled weak images [n_l, 3, H, W]
        img_u_w: Unlabeled weak images [n_u, 3, H, W]
        n_l: Number of labeled samples
        hw: Target (H, W) size

    Returns:
        h_w_: Feature map size
        pseudo_logit: Pseudo label confidence [n_u, H, W]
        pseudo_label: Pseudo label predictions [n_u, H, W]
        t_entropy_all: Entropy map [n_l+n_u, H, W]
    """
    with torch.no_grad():
        t_pred_all = model_ema(torch.cat((img_l_w, img_u_w), dim=0))  # [2n, C, H', W']
        h_w_ = t_pred_all.shape[-2:]

        t_pred_all = F.interpolate(
            t_pred_all, size=hw, mode="bilinear", align_corners=True
        )  # [2n, C, H, W]
        t_prob_all = torch.softmax(t_pred_all, dim=1)  # [2n, C, H, W]

        pseudo_logit, pseudo_label = torch.max(t_prob_all[n_l:], dim=1)  # [n_u, H, W]
        t_entropy_all = torch.sum(
            -t_prob_all * torch.log(t_prob_all + 1e-8), dim=1
        )  # [2n, H, W]

    return h_w_, pseudo_logit, pseudo_label, t_entropy_all


def main(args, cfg):
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), **SEGMIND_CONFIG, "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Model configurations for different backbone sizes
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
    patch_size = 14 if backbone_version == "dinov2" else 16

    # Initialize model
    model = DPT_SegMind(
        **{**model_configs[backbone_size], "nclass": cfg["nclass"]},
        backbone_version=backbone_version,
    )

    # Load pretrained backbone weights
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)

    if cfg["lock_backbone"]:
        model.lock_backbone()

    # Optimizer with different learning rates for backbone and head
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

    # EMA teacher model
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    # Loss functions
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
    criterion_e = nn.MSELoss().cuda(local_rank)  # Entropy distillation loss
    criterion_r = nn.MSELoss().cuda(local_rank)  # Reconstruction loss
    criterion_rsc = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"]).cuda(
        local_rank
    )  # Reconstruction segmentation loss

    # Datasets
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

    # Data loaders
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

    # Initialize memory bank for contrastive learning
    memory_bank_list, queue_size, queue_ptr_list = init_memory_bank(
        cfg["nclass"],
        bank_size=SEGMIND_CONFIG["bank_size"],
        feat_dim=256,
    )

    # Resume from checkpoint
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

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

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
        total_loss_e = AverageMeter()
        total_loss_r = AverageMeter()
        total_loss_c = AverageMeter()
        total_mask_ratio = AverageMeter()

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

            n_l = img_x.shape[0]
            hw = mask_x.shape[-2:]

            # Get pseudo labels from EMA teacher
            h_w_, pseudo_logit, pseudo_label, t_entropy_all = get_pseudo_labels(
                model_ema, img_x, img_u_w, n_l, hw
            )

            # ClassMix augmentation for unlabeled data
            img_u_w, img_u_s, pseudo_label, pseudo_logit, t_entropy_u = generate_u_data(
                img_u_w,
                img_u_s,
                pseudo_label,
                pseudo_logit,
                t_entropy_all[n_l:],
                device=img_u_w.device,
            )

            # Combine labels
            lab_all = torch.cat((mask_x, pseudo_label), dim=0)
            lab_u_reli = pseudo_logit.ge(SEGMIND_CONFIG["conf_thresh"]).float()
            mask_all = torch.cat((mask_x >= 0, lab_u_reli), dim=0)

            # CutMix for strong augmented images
            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = img_u_s.flip(
                0
            )[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

            # Combine all images for student prediction
            img_all_s = torch.cat((img_x, img_u_s), dim=0)
            img_all_w = torch.cat((img_x, img_u_w), dim=0)

            # Determine if we need reconstruction mode
            need_recon = (
                SEGMIND_CONFIG["lambda_r"] > 0 or SEGMIND_CONFIG["lambda_rsc"] > 0
            ) and epoch <= SEGMIND_CONFIG["epoch_pre"]
            need_contrastive = SEGMIND_CONFIG["lambda_c"] > 0

            # Generate mask for reconstruction if needed
            if need_recon or need_contrastive:
                nchw = img_all_s.size()
                mask_tensor = get_batch_mask_tensor(
                    nchw=nchw, mask_rate=SEGMIND_CONFIG["mask_rate"]
                ).cuda()

                # Single forward with reconstruction mode to get all outputs
                # Apply mask only for reconstruction input, but we use strong augmented images
                masked_img = img_all_s * mask_tensor if need_recon else img_all_s
                s_pred_all, s_feat_all, r_img_all = model(
                    masked_img, mode="r", mask=mask_tensor
                )
            else:
                # Standard forward without reconstruction
                s_pred_all = model(img_all_s)
                s_feat_all = None
                r_img_all = None
                mask_tensor = None

            s_pred_all = F.interpolate(
                s_pred_all, size=hw, mode="bilinear", align_corners=True
            )
            s_prob_all = torch.softmax(s_pred_all, dim=1)

            # Split predictions
            pred_x, pred_u_s = s_pred_all.split([n_l, img_u_s.shape[0]])

            # ============================================================
            # Loss Computation
            # ============================================================

            # 1. Supervised loss (loss_l)
            loss_l = criterion_l(pred_x, mask_x) * SEGMIND_CONFIG["lambda_l"]

            # 2. Unsupervised consistency loss (loss_u / loss_s)
            mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = (
                pseudo_label.clone(),
                pseudo_logit.clone(),
                ignore_mask.clone(),
            )

            mask_u_w_cutmixed[cutmix_box == 1] = pseudo_label.flip(0)[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = pseudo_logit.flip(0)[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask.flip(0)[cutmix_box == 1]

            loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed)
            loss_mask = (conf_u_w_cutmixed >= SEGMIND_CONFIG["conf_thresh"]) & (
                ignore_mask_cutmixed != 255
            )
            loss_u_s = (loss_u_s * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

            # 3. Entropy distillation loss (loss_e)
            loss_e = torch.tensor(0.0).cuda()
            if SEGMIND_CONFIG["lambda_e"] > 0:
                s_entropy_all = torch.sum(
                    -s_prob_all * torch.log(s_prob_all + 1e-8), dim=1
                )
                t_entropy_combined = torch.cat(
                    (t_entropy_all[:n_l], t_entropy_u), dim=0
                )
                loss_e = (
                    criterion_e(s_entropy_all, t_entropy_combined)
                    * SEGMIND_CONFIG["lambda_e"]
                )

            # 4. Reconstruction loss (loss_r, loss_rsc) - only for first epoch_pre epochs
            loss_r = torch.tensor(0.0).cuda()
            loss_rsc = torch.tensor(0.0).cuda()
            if need_recon and r_img_all is not None:
                r_img_all = F.interpolate(
                    r_img_all, size=hw, mode="bilinear", align_corners=True
                )

                if SEGMIND_CONFIG["lambda_r"] > 0:
                    # Reconstruction loss on masked regions
                    loss_r = (
                        criterion_r(
                            r_img_all.permute(0, 2, 3, 1)[
                                ~mask_tensor.bool().squeeze(1)
                            ],
                            img_all_s[:, :3, :, :].permute(0, 2, 3, 1)[
                                ~mask_tensor.bool().squeeze(1)
                            ],
                        )
                        * SEGMIND_CONFIG["lambda_r"]
                    )

                if SEGMIND_CONFIG["lambda_rsc"] > 0:
                    # Reconstruction segmentation loss
                    r_pred_interp = F.interpolate(
                        s_pred_all, size=hw, mode="bilinear", align_corners=True
                    )
                    loss_rsc = (
                        criterion_rsc(
                            r_pred_interp.permute(0, 2, 3, 1)[
                                ~mask_tensor.bool().squeeze(1)
                            ],
                            lab_all[~mask_tensor.bool().squeeze(1)],
                        )
                        * SEGMIND_CONFIG["lambda_rsc"]
                    )

            # 5. Contrastive loss (loss_c)
            loss_c = torch.tensor(0.0).cuda()
            if need_contrastive and s_feat_all is not None:
                s_feat_small = F.interpolate(
                    s_feat_all, size=h_w_, mode="bilinear", align_corners=True
                )
                with torch.no_grad():
                    lab_all_small = F.interpolate(
                        lab_all.float().unsqueeze(1), size=h_w_, mode="nearest"
                    ).squeeze(1)
                    s_prob_small = F.interpolate(
                        s_prob_all.detach(),
                        size=h_w_,
                        mode="bilinear",
                        align_corners=True,
                    )

                loss_c = (
                    cal_c_loss(
                        feat=s_feat_small,
                        lab=lab_all_small.long(),
                        prob=s_prob_small,
                        class_num=cfg["nclass"],
                        memory_bank_list=memory_bank_list,
                        queue_size=queue_size,
                        queue_ptr_list=queue_ptr_list,
                        query_threshold=SEGMIND_CONFIG["query_threshold"],
                        temperature=SEGMIND_CONFIG["temperature"],
                        num_query=SEGMIND_CONFIG["num_query"],
                        num_negative=SEGMIND_CONFIG["num_negative"],
                        device=s_feat_small.device,
                    )
                    * SEGMIND_CONFIG["lambda_c"]
                )

            # Total loss
            loss = (loss_l + loss_u_s + loss_e + loss_r + loss_rsc + loss_c) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update meters
            total_loss.update(loss.item())
            total_loss_x.update(loss_l.item())
            total_loss_s.update(loss_u_s.item())
            total_loss_e.update(loss_e.item())
            total_loss_r.update((loss_r + loss_rsc).item())
            total_loss_c.update(loss_c.item())
            mask_ratio = (
                (pseudo_logit >= SEGMIND_CONFIG["conf_thresh"]) & (ignore_mask != 255)
            ).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            # Learning rate schedule
            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            # EMA update
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
                writer.add_scalar("train/loss_x", loss_l.item(), iters)
                writer.add_scalar("train/loss_s", loss_u_s.item(), iters)
                writer.add_scalar("train/loss_e", loss_e.item(), iters)
                writer.add_scalar("train/loss_r", (loss_r + loss_rsc).item(), iters)
                writer.add_scalar("train/loss_c", loss_c.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total: {:.3f}, L_x: {:.3f}, L_s: {:.3f}, "
                    "L_e: {:.3f}, L_r: {:.3f}, L_c: {:.3f}, Mask: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_loss_e.avg,
                        total_loss_r.avg,
                        total_loss_c.avg,
                        total_mask_ratio.avg,
                    )
                )

        # Validation
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
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
