"""
ScaleMatch: Multi-scale semi-supervised semantic segmentation.

Ported from the official ScaleMatch training logic onto the SemiFT DPT backbone.
"""

import argparse
import logging
import os
import pprint
import random
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset as NaturalSemiDataset
from dataset.semi_rs import SemiDataset as RemoteSemiDataset
from dataset.val import ValDataset
from model.semseg.dpt_scalematch import DPT_ScaleMatch
from supervised import validation_cpu
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.focal import FocalLoss
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.train_utils import (
    DictAverageMeter,
    confidence_weighted_loss,
    cutmix_img_,
    cutmix_mask,
)


NATURAL_IMAGE_DATASETS = {"pascal", "cityscapes"}
REMOTE_SENSING_DATASETS = {"iSAID", "vaihingen", "potsdam", "loveda"}


def get_scalematch_dataset_cls(dataset_name):
    if dataset_name in NATURAL_IMAGE_DATASETS:
        return NaturalSemiDataset, "semi"
    if dataset_name in REMOTE_SENSING_DATASETS:
        return RemoteSemiDataset, "semi_rs"
    raise ValueError(
        f"Unsupported dataset for scalematch: {dataset_name}. "
        f"Please register it in NATURAL_IMAGE_DATASETS or REMOTE_SENSING_DATASETS."
    )


def get_parser():
    parser = argparse.ArgumentParser(
        description="ScaleMatch: Multi-scale Semi-Supervised Semantic Segmentation"
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
    amp = cfg.get("amp", False)

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Model configurations
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
    # Initialize model
    model = DPT_ScaleMatch(
        **{**model_configs[backbone_size], "nclass": cfg["nclass"]},
        backbone_version=backbone_version,
    )

    # Load pretrained backbone
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)

    if cfg["lock_backbone"]:
        model.lock_backbone()

    optimizer = SGD(
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
        momentum=0.9,
        weight_decay=1e-4,
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
        find_unused_parameters=False,
    )

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

    criterion_u = nn.CrossEntropyLoss(
        reduction="none", ignore_index=cfg.get("ignore_index", 255)
    ).cuda(local_rank)

    # Datasets
    SemiDataset, dataset_loader_name = get_scalematch_dataset_cls(cfg["dataset"])
    if rank == 0:
        logger.info(f"ScaleMatch dataset loader: {dataset_loader_name} for {cfg['dataset']}")

    trainset_u = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        ignore_index=cfg.get("ignore_index", 255),
    )
    trainset_l = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(trainset_u.ids),
        ignore_index=cfg.get("ignore_index", 255),
    )
    valset = ValDataset(
        cfg["dataset"],
        cfg["data_root"],
        "val",
        ignore_value=cfg.get("ignore_index", 255),
    )

    # Data loaders
    # if ddp:
    #     trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    #     trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    #     valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    # else:
    #     trainsampler_l = None
    #     trainsampler_u = None
    #     valsampler = None

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

    # ScaleMatch specific configs
    img_scales = cfg.get("img_scales", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    feat_s_scales = cfg.get("feat_s_scales", [0.5, 0.75, 1.0])
    feat_l_scales = cfg.get("feat_l_scales", [1.0, 1.25, 1.5])
    conf_thresh = cfg.get("conf_thresh", 0.95)
    warm_up = cfg.get("warm_up", 5)

    total_epochs = cfg["epochs"]
    total_iters = len(trainloader_u) * total_epochs
    previous_best = 0.0
    best_epoch = 0
    epoch = -1
    ETA = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Resume from checkpoint
    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(
            os.path.join(args.save_path, "latest.pth"), map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        best_epoch = checkpoint.get("best_epoch", 0)

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    is_best = False
    ignore_index = cfg.get("ignore_index", 255)

    for epoch in range(epoch + 1, total_epochs):
        start_time = time.time()

        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f} @epoch-{:}, ETA: {:.2f}M".format(
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    previous_best,
                    best_epoch,
                    ETA / 60,
                )
            )

        log_avg = DictAverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        model.train()

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
            (img_u_w_mix, img_u_s1_mix, _, ignore_mask_mix, _, _),
        ) in enumerate(loader):
            t0 = time.time()

            random_scale = random.choice(img_scales)
            feature_scale = random.choice(
                feat_s_scales if random_scale > 1 else feat_l_scales
            )

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, ignore_mask = img_u_s1.cuda(), ignore_mask.cuda()
            cutmix_box1 = cutmix_box1.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix = img_u_s1_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            iters = epoch * len(trainloader_u) + i
            cutmix_img_(img_u_s1, img_u_s1_mix, cutmix_box1)

            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp):
                    pred_u_w_mix = model.module(img_u_w_mix, scale_factor=None)
                    if isinstance(pred_u_w_mix, dict):
                        pred_u_w_mix = pred_u_w_mix["pred_ori"]
                    conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)
            model.train()

            num_lb = img_x.shape[0]

            optimizer.zero_grad()
            with model.no_sync():
                with torch.cuda.amp.autocast(enabled=amp):
                    pred = model(
                        torch.cat((img_x, img_u_w)),
                        scale_factor=random_scale,
                        feature_scale=feature_scale,
                    )

                    if epoch < warm_up:
                        pred_u_w = pred["pred_ori"][num_lb:]
                    else:
                        pred_u_w = pred["pred_joint"][num_lb:]

                    pred_u_w = pred_u_w.detach()
                    conf_u_w, mask_u_w = pred_u_w.softmax(dim=1).max(dim=1)

                    mask_u_w_cutmixed1 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box1)
                    conf_u_w_cutmixed1 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box1)
                    ignore_mask_cutmixed1 = cutmix_mask(
                        ignore_mask, ignore_mask_mix, cutmix_box1
                    )

                    pred_x_joint = pred["pred_joint"][:num_lb]
                    pred_u_w_scale = pred["pred_size"][num_lb:]
                    pred_u_w_fp = pred["pred_fp"][num_lb:]

                    loss_x = criterion_l(pred_x_joint, mask_x)

                    loss_u_size = criterion_u(pred_u_w_scale, mask_u_w)
                    loss_u_size = confidence_weighted_loss(
                        loss_u_size,
                        conf_u_w,
                        ignore_mask,
                        ignore_index,
                        conf_thresh=conf_thresh,
                    )

                    loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                    loss_u_w_fp = confidence_weighted_loss(
                        loss_u_w_fp,
                        conf_u_w,
                        ignore_mask,
                        ignore_index,
                        conf_thresh=conf_thresh,
                    )

                    loss_part1 = (loss_x + 0.25 * loss_u_size + 0.5 * loss_u_w_fp) / 2.0

                scaler.scale(loss_part1).backward()

            with torch.cuda.amp.autocast(enabled=amp):
                pred_u_s = model(img_u_s1, scale_factor=None)
                if isinstance(pred_u_s, dict):
                    pred_u_s = pred_u_s["pred_ori"]

                loss_u_s1 = criterion_u(pred_u_s, mask_u_w_cutmixed1)
                loss_u_s1 = confidence_weighted_loss(
                    loss_u_s1,
                    conf_u_w_cutmixed1,
                    ignore_mask_cutmixed1,
                    ignore_index,
                    conf_thresh=conf_thresh,
                )

                loss_part2 = (0.25 * loss_u_s1) / 2.0
                total_loss = (
                    loss_x
                    + 0.25 * loss_u_s1
                    + 0.25 * loss_u_size
                    + 0.5 * loss_u_w_fp
                ) / 2.0

            scaler.scale(loss_part2).backward()
            scaler.step(optimizer)
            scaler.update()

            valid_mask = ignore_mask != ignore_index
            mask_ratio = (
                ((conf_u_w >= conf_thresh) & valid_mask).sum().float()
                / valid_mask.sum().clamp(min=1.0)
            )

            log_avg.update(
                {
                    "iter_time": time.time() - t0,
                    "Total_loss": total_loss,
                    "Loss_x": loss_x,
                    "Loss_u_s": loss_u_s1,
                    "Loss_u_scale": loss_u_size,
                    "Loss_u_fp": loss_u_w_fp,
                    "Mask_ratio": mask_ratio,
                }
            )

            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                for k, v in log_avg.avgs.items():
                    writer.add_scalar("train/" + k, v.item() if torch.is_tensor(v) else v, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(f"Iters: {i}, " + str(log_avg))
                log_avg.reset()

        eval_mode = "sliding_window" if cfg["dataset"] == "cityscapes" else "original"
        mIoU, iou_class = validation_cpu(cfg, model, valloader)

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
            for i, iou in enumerate(iou_class):
                writer.add_scalar(
                    "eval/%s_IoU" % CLASSES[cfg["dataset"]][i], iou, epoch
                )

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if mIoU == previous_best:
            best_epoch = epoch

        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
                "best_epoch": best_epoch,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))

        end_time = time.time()
        ETA = (total_epochs - (epoch + 1)) * (end_time - start_time)


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
