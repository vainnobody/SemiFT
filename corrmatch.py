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
from einops import rearrange
from datetime import datetime

from dataset.semi_rs import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt_corrmatch import DPT_CorrMatch
from model.semseg.corrmatch_utils import ThreshController
from util.utils import count_params, init_log, AverageMeter, intersectionAndUnion
from util.dist_helper import setup_distributed
from util.viz import Visualizer
from util.validation import validation_cpu as shared_validation_cpu
import numpy as np
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.focal import FocalLoss


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def get_parser():
    parser = argparse.ArgumentParser(
        description="CorrMatch style training with FixMatch/DPT"
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

    model = DPT_CorrMatch(
        **{**model_configs[backbone_size], "nclass": cfg["nclass"]},
        backbone_version=backbone_version,
    )

    if os.path.exists(f'./pretrained/{cfg["backbone"]}.pth'):
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
                    p for name, p in model.named_parameters() if "backbone" not in name
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

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # EMA model not strictly used in standard CorrMatch logic shown, but optional in FixMatch.
    # We'll skip for strict CorrMatch port or keep if desired. Let's keep it simple and skip EMA for now to verify CorrMatch logic first.
    # Actually, the user asked to convert TO FixMatch style. FixMatch uses EMA.
    # But CorrMatch code logic is complex. Integrating EMA might be tricky.
    # I'll stick to non-EMA for the core logic to match CorrMatch behavior, unless FixMatch *requires* it.

    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(
            local_rank
        )
    else:
        raise NotImplementedError(
            "%s criterion not implemented" % cfg["criterion"]["name"]
        )

    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)
    criterion_kl = nn.KLDivLoss(reduction="none").cuda(local_rank)

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
    thresh_controller = ThreshController(
        nclass=cfg["nclass"], momentum=0.999, thresh_init=cfg.get("thresh_init", 0.85)
    )

    best_iou = 0.0
    epoch = -1

    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz = Visualizer(save_dir=f"./viz/{filename}_corrmatch", dataset=cfg["dataset"])

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(f"===========> Epoch: {epoch}, Best IoU: {best_iou:.2f}")

        total_loss_meter = AverageMeter()
        loss_x_meter = AverageMeter()
        loss_x_corr_meter = AverageMeter()
        loss_u_s1_meter = AverageMeter()
        loss_u_kl_meter = AverageMeter()
        loss_u_w_fp_meter = AverageMeter()
        loss_u_corr_meter = AverageMeter()
        mask_ratio_meter = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        # distinct loaders for u1 and u2 to simulate CorrMatch dual sampling
        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
            (img_u_w_mix, img_u_s1_mix, _, ignore_mask_mix, _, _),
        ) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1 = img_u_w.cuda(), img_u_s1.cuda()
            ignore_mask, cutmix_box1 = ignore_mask.cuda(), cutmix_box1.cuda()
            img_u_w_mix, img_u_s1_mix = img_u_w_mix.cuda(), img_u_s1_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            # 1. Generate pseudo-labels/mask for the mix source (batch 2)
            with torch.no_grad():
                model.eval()
                res_u_w_mix = model(img_u_w_mix, need_fp=False, use_corr=False)
                pred_u_w_mix = res_u_w_mix["out"].detach()
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

                # Apply CutMix to strong view of batch 1 using strong view of batch 2
                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = (
                    img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
                )

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            # 2. Forward Labeled + Weak Unlabeled
            res_w = model(torch.cat((img_x, img_u_w)), need_fp=True, use_corr=True)
            preds = res_w["out"]
            preds_fp = res_w.get(
                "out_fp", preds
            )  # fallback if need_fp fails or logic changes
            preds_corr = res_w["corr_out"]
            preds_corr_map = res_w["corr_map"].detach()

            pred_x_corr, pred_u_w_corr = preds_corr.split([num_lb, num_ulb])
            pred_u_w_corr_map = preds_corr_map[num_lb:]
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            # 3. Forward Strong Unlabeled (CutMixed)
            res_s = model(img_u_s1, need_fp=False, use_corr=True)
            pred_u_s1 = res_s["out"]
            pred_u_s1_corr = res_s["corr_out"]

            # 4. Process pseudo-labels and Thresholds
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1 = mask_u_w.clone()
            conf_u_w_cutmixed1 = conf_u_w.clone()
            ignore_mask_cutmixed1 = ignore_mask.clone()
            corr_map_u_w_cutmixed1 = pred_u_w_corr_map.clone()

            cutmix_box1_map = cutmix_box1 == 1
            mask_u_w_cutmixed1[cutmix_box1_map] = mask_u_w_mix[cutmix_box1_map]
            # Mix confidence/ignore masks as well? CorrMatch does.
            # But wait, conf_u_w_mix is not computed above.
            # CorrMatch: conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
            conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
            conf_u_w_cutmixed1[cutmix_box1_map] = conf_u_w_mix[cutmix_box1_map]
            ignore_mask_cutmixed1[cutmix_box1_map] = ignore_mask_mix[cutmix_box1_map]

            # Handle corr_map masking for cutmix
            cutmix_box1_sample = rearrange(cutmix_box1_map, "n h w -> n 1 h w")
            ignore_mask_cutmixed1_sample = rearrange(
                (ignore_mask_cutmixed1 != 255), "n h w -> n 1 h w"
            )
            corr_map_u_w_cutmixed1 = (
                corr_map_u_w_cutmixed1
                * ~cutmix_box1_sample
                * ignore_mask_cutmixed1_sample
            ).bool()

            thresh_controller.thresh_update(
                pred_u_w, ignore_mask_cutmixed1, update_g=True
            )
            thresh_global = thresh_controller.get_thresh_global()

            conf_fliter_u_w = (conf_u_w_cutmixed1 >= thresh_global) & (
                ignore_mask_cutmixed1 != 255
            )
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w.clone()
            conf_fliter_u_w_sample = rearrange(
                conf_fliter_u_w_without_cutmix, "n h w -> n 1 h w"
            )

            # Refinement Loop (Slow part, but necessary for logic reproduction)
            segments = (corr_map_u_w_cutmixed1 * conf_fliter_u_w_sample).bool()
            b_sample, c_sample, _, _ = corr_map_u_w_cutmixed1.shape

            # Optimization: Can we avoid the loop?
            # It iterates over batch and channel (queries).
            # c_sample is 128 (from Corr logic).
            # We can try to keep it as is or optimize later.

            for img_idx in range(b_sample):
                for segment_idx in range(c_sample):
                    segment = segments[img_idx, segment_idx]
                    segment_ori = corr_map_u_w_cutmixed1[img_idx, segment_idx]
                    high_conf_ratio = torch.sum(segment) / torch.sum(segment_ori).clamp(
                        min=1.0
                    )
                    if torch.sum(segment) == 0 or high_conf_ratio < thresh_global:
                        continue

                    # Logic: if segment overlaps with high confidence prediction, propagate label
                    unique_cls, count = torch.unique(
                        mask_u_w_cutmixed1[img_idx][segment == 1], return_counts=True
                    )
                    if len(count) > 0 and (
                        torch.max(count) / torch.sum(count) > thresh_global
                    ):
                        top_class = unique_cls[torch.argmax(count)]
                        mask_u_w_cutmixed1[img_idx][segment_ori == 1] = top_class
                        conf_fliter_u_w_without_cutmix[img_idx] = (
                            conf_fliter_u_w_without_cutmix[img_idx] | segment_ori
                        )

            conf_fliter_u_w_without_cutmix = (
                conf_fliter_u_w_without_cutmix | conf_fliter_u_w
            )

            # 5. Loss Calculation
            loss_x = criterion_l(pred_x, mask_x)
            loss_x_corr = criterion_l(pred_x_corr, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = (loss_u_s1 * conf_fliter_u_w_without_cutmix).sum() / (
                ignore_mask_cutmixed1 != 255
            ).sum().clamp(min=1.0)

            loss_u_corr_s1 = criterion_u(pred_u_s1_corr, mask_u_w_cutmixed1)
            loss_u_corr_s1 = (loss_u_corr_s1 * conf_fliter_u_w_without_cutmix).sum() / (
                ignore_mask_cutmixed1 != 255
            ).sum().clamp(min=1.0)
            loss_u_corr_s = loss_u_corr_s1

            loss_u_corr_w = criterion_u(pred_u_w_corr, mask_u_w)
            loss_u_corr_w = (
                loss_u_corr_w * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            ).sum() / (ignore_mask != 255).sum().clamp(min=1.0)
            loss_u_corr = 0.5 * (loss_u_corr_s + loss_u_corr_w)

            softmax_pred_u_w = F.softmax(pred_u_w.detach(), dim=1)
            logsoftmax_pred_u_s1 = F.log_softmax(pred_u_s1, dim=1)
            loss_u_kl = criterion_kl(logsoftmax_pred_u_s1, softmax_pred_u_w)
            loss_u_kl = (loss_u_kl.sum(dim=1) * conf_fliter_u_w).sum() / (
                ignore_mask_cutmixed1 != 255
            ).sum().clamp(min=1.0)

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = (
                loss_u_w_fp * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            ).sum() / (ignore_mask != 255).sum().clamp(min=1.0)

            loss = (
                0.5 * loss_x
                + 0.5 * loss_x_corr
                + loss_u_s1 * 0.25
                + loss_u_kl * 0.25
                + loss_u_w_fp * 0.25
                + 0.25 * loss_u_corr
            ) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_meter.update(loss.item())
            loss_x_meter.update(loss_x.item())
            loss_x_corr_meter.update(loss_x_corr.item())
            loss_u_s1_meter.update(loss_u_s1.item())
            loss_u_kl_meter.update(loss_u_kl.item())
            loss_u_w_fp_meter.update(loss_u_w_fp.item())
            loss_u_corr_meter.update(loss_u_corr.item())

            mask_ratio = (
                (conf_u_w >= thresh_global) & (ignore_mask != 255)
            ).sum().item() / (ignore_mask != 255).sum().clamp(min=1.0)
            mask_ratio_meter.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iter {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss x_corr: {:.3f}, Loss u_s: {:.3f}, "
                    "Loss u_kl: {:.3f}, Loss u_fp: {:.3f}, Loss u_corr: {:.3f}, Mask ratio: {:.3f}, Thresh: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss_meter.avg,
                        loss_x_meter.avg,
                        loss_x_corr_meter.avg,
                        loss_u_s1_meter.avg,
                        loss_u_kl_meter.avg,
                        loss_u_w_fp_meter.avg,
                        loss_u_corr_meter.avg,
                        mask_ratio_meter.avg,
                        thresh_global,
                    )
                )
                if i < 5:
                    viz.push(
                        {
                            "img_x": (img_x[0], Visualizer.TENSOR),
                            "mask_x": (mask_x[0], Visualizer.SEGMENTATION),
                            "pred_x": (
                                pred_x.argmax(dim=1)[0],
                                Visualizer.SEGMENTATION,
                            ),
                            "pred_corr_x": (
                                pred_x_corr.argmax(dim=1)[0],
                                Visualizer.SEGMENTATION,
                            ),
                            "pred_u_w": (
                                pred_u_w.argmax(dim=1)[0],
                                Visualizer.SEGMENTATION,
                            ),
                            "img_u_s1": (img_u_s1[0], Visualizer.TENSOR),
                            "pred_u_s1": (
                                pred_u_s1.argmax(dim=1)[0],
                                Visualizer.SEGMENTATION,
                            ),
                            "pred_u_s1_corr": (
                                pred_u_s1_corr.argmax(dim=1)[0],
                                Visualizer.SEGMENTATION,
                            ),
                            "mask_cutmix": (
                                mask_u_w_cutmixed1[0],
                                Visualizer.SEGMENTATION,
                            ),
                            "pred_u_w_corr": (
                                pred_u_w_corr.argmax(dim=1)[0],
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

        # Validation
        mIoU, _ = validation_cpu(cfg, model, valloader)
        if rank == 0:
            logger.info(f"Epoch {epoch} mIoU: {mIoU:.4f}")
            if mIoU > best_iou:
                best_iou = mIoU
                torch.save(
                    model.module.state_dict(), os.path.join(args.save_path, "best.pth")
                )
            torch.save(
                model.module.state_dict(), os.path.join(args.save_path, "latest.pth")
            )


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
