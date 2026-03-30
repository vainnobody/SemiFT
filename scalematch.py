import argparse
import logging
import os
import pprint
import random

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi_rs import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.upernet import UperNet
from model.semseg.scalematch import ScaleMatchModel
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.focal import FocalLoss
from util.train_utils import DictAverageMeter, confidence_weighted_loss, cutmix_img_, cutmix_mask
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
from util.ssl_method_utils import (
    get_local_rank,
    load_checkpoint_on_cpu,
    save_checkpoint_to_disk,
    log_cuda_memory,
    get_model_kwargs,
    get_backbone_info,
    load_backbone_checkpoint,
)
from util.validation import validation_cpu as shared_validation_cpu


DEFAULT_IMG_SCALES = [0.25, 0.5, 1.5, 2.0]
DEFAULT_FEAT_S_SCALES = [0.75]
DEFAULT_FEAT_L_SCALES = [1.25]
OFFICIAL_WARM_UP = 10
OFFICIAL_CONF_THRESH = 0.0
OFFICIAL_USE_AMP = False


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def get_parser():
    parser = argparse.ArgumentParser(
        description="ScaleMatch port based on official GitHub recipe and SemiFT supervised.py runtime"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def build_scalematch_model(cfg):
    model_kwargs = get_model_kwargs(cfg)
    _, backbone_version = get_backbone_info(cfg)

    if cfg["model"] == "dpt":
        base_model = DPT(
            **model_kwargs,
            backbone_version=backbone_version,
        )
    elif cfg["model"] == "upernet":
        base_model = UperNet(
            **model_kwargs,
            backbone_version=backbone_version,
        )
    else:
        raise ValueError(f'Unsupported model type: {cfg["model"]}')

    load_backbone_checkpoint(base_model, cfg)
    model = ScaleMatchModel(base_model, cfg["nclass"])

    if cfg.get("lock_backbone", False):
        model.lock_backbone()

    return model


def unpack_unlabeled_batch(batch):
    if len(batch) == 6:
        return (*batch, None)
    if len(batch) == 7:
        return batch
    raise ValueError(f"Unexpected unlabeled batch size: {len(batch)}")


def main(args, cfg):
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    ignore_index = cfg.get("ignore_index", 255)
    amp = OFFICIAL_USE_AMP
    img_scales = DEFAULT_IMG_SCALES
    feat_s_scales = DEFAULT_FEAT_S_SCALES
    feat_l_scales = DEFAULT_FEAT_L_SCALES
    warm_up = OFFICIAL_WARM_UP
    conf_thresh = OFFICIAL_CONF_THRESH

    if rank == 0:
        all_args = {
            **cfg,
            **vars(args),
            "ngpus": world_size,
            "scalematch_img_scales": img_scales,
            "scalematch_feat_s_scales": feat_s_scales,
            "scalematch_feat_l_scales": feat_l_scales,
            "scalematch_warm_up": warm_up,
            "scalematch_conf_thresh": conf_thresh,
            "scalematch_amp": amp,
        }
        logger.info("{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = build_scalematch_model(cfg)
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

    local_rank = get_local_rank()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    log_cuda_memory(
        logger,
        rank,
        "after_model_to_cuda",
        local_rank=local_rank,
        save_path=args.save_path,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )
    if hasattr(model, "_set_static_graph"):
        model._set_static_graph()
        if rank == 0:
            logger.info("Enabled DDP static graph for ScaleMatch training.")
    log_cuda_memory(
        logger, rank, "after_ddp_wrap", local_rank=local_rank, save_path=args.save_path
    )

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

    trainsampler_u_mix = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u_mix = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_u_mix,
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
    previous_best = 0.0
    best_epoch = -1
    epoch = -1
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        log_cuda_memory(logger, rank, "before_resume_load", save_path=args.save_path)
        checkpoint = load_checkpoint_on_cpu(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        log_cuda_memory(logger, rank, "after_resume_load", save_path=args.save_path)
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        best_epoch = checkpoint.get("best_epoch", -1)

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}".format(
                    epoch, previous_best, best_epoch
                )
            )

        log_avg = DictAverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        trainloader_u_mix.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)
        model.train()

        for i, (
            (img_x, mask_x),
            batch_u,
            batch_u_mix,
        ) in enumerate(loader):
            (
                img_u_w,
                img_u_s1,
                img_u_s2,
                ignore_mask,
                cutmix_box1,
                cutmix_box2,
                _,
            ) = unpack_unlabeled_batch(batch_u)
            (
                img_u_w_mix,
                img_u_s1_mix,
                img_u_s2_mix,
                ignore_mask_mix,
                _,
                _,
                _,
            ) = unpack_unlabeled_batch(batch_u_mix)

            random_scale = random.choice(img_scales)
            feature_scale = random.choice(
                feat_s_scales if random_scale > 1 else feat_l_scales
            )

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = (
                img_u_s1.cuda(),
                img_u_s2.cuda(),
                ignore_mask.cuda(),
            )
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            cutmix_img_(img_u_s1, img_u_s1_mix, cutmix_box1)
            cutmix_img_(img_u_s2, img_u_s2_mix, cutmix_box2)

            with torch.cuda.amp.autocast(enabled=amp):
                model.eval()
                with torch.no_grad():
                    pred_u_w_mix = model(img_u_w_mix, scale_factor=None, scales=None)
                conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)

                model.train()

                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
                pred = model(
                    torch.cat((img_x, img_u_w)),
                    scale_factor=random_scale,
                    feature_scale=feature_scale,
                )
                pred_u_s = model(img_u_s1, scale_factor=None, scales=None)

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

                loss_u_s1 = criterion_u(pred_u_s, mask_u_w_cutmixed1)
                loss_u_s1 = confidence_weighted_loss(
                    loss_u_s1,
                    conf_u_w_cutmixed1,
                    ignore_mask_cutmixed1,
                    ignore_index=ignore_index,
                    conf_thresh=conf_thresh,
                )
                loss_u_size = criterion_u(pred_u_w_scale, mask_u_w)
                loss_u_size = confidence_weighted_loss(
                    loss_u_size,
                    conf_u_w,
                    ignore_mask,
                    ignore_index=ignore_index,
                    conf_thresh=conf_thresh,
                )
                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = confidence_weighted_loss(
                    loss_u_w_fp,
                    conf_u_w,
                    ignore_mask,
                    ignore_index=ignore_index,
                    conf_thresh=conf_thresh,
                )

                loss_standard = loss_u_s1 * 0.25 + loss_u_size * 0.25 + loss_u_w_fp * 0.5
                total_loss = (loss_x + loss_standard) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            if amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            valid_mask = (ignore_mask != ignore_index)
            mask_ratio = (
                ((conf_u_w >= conf_thresh) & valid_mask).sum().item()
                / valid_mask.sum().clamp(min=1).item()
            )

            log_avg.update(
                {
                    "Total loss": total_loss,
                    "Loss x": loss_x,
                    "Loss u_s": loss_u_s1,
                    "Loss u_scale": loss_u_size,
                    "Loss w_fp_scale": loss_u_w_fp,
                    "Mask ratio": mask_ratio,
                }
            )

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                for k, v in log_avg.avgs.items():
                    writer.add_scalar("train/" + k, v, iters)

            if (i % max(1, len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(f"Iters: {i}, " + str(log_avg))
                log_avg.reset()

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode", "slide_window" if cfg["dataset"] == "cityscapes" else "original"
        )
        val_cfg.setdefault("ignore_index", ignore_index)
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
            for i, iou in enumerate(iou_class):
                writer.add_scalar(
                    "eval/%s_IoU" % (CLASSES[cfg["dataset"]][i]), iou, epoch
                )

        is_best = mIoU >= previous_best
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
            save_checkpoint_to_disk(
                checkpoint,
                os.path.join(args.save_path, "latest.pth"),
                os.path.join(args.save_path, "best.pth"),
                is_best=is_best,
            )


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg.setdefault("ignore_index", 255)
    main(args, cfg)
