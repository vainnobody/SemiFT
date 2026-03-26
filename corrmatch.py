import argparse
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

from dataset.semi_rs import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.upernet import UperNet
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.focal import FocalLoss
from util.utils import count_params, init_log, AverageMeter
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
from model.semseg.corrmatch_utils import ThreshController


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def get_parser():
    parser = argparse.ArgumentParser(
        description="CorrMatch training rebuilt on top of the supervised.py scaffold"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def build_model(cfg, backbone_version):
    model_kwargs = get_model_kwargs(cfg)

    if cfg["model"] == "dpt":
        model = DPT(
            **model_kwargs,
            backbone_version=backbone_version,
            enable_corrmatch=True,
        )
    elif cfg["model"] == "upernet":
        model = UperNet(
            **model_kwargs,
            backbone_version=backbone_version,
            enable_corrmatch=True,
        )
    else:
        raise ValueError(f"Unsupported CorrMatch model: {cfg['model']}")

    return model


def refine_corr_pseudo_labels(
    corr_map_u_w_cutmixed1,
    conf_filter_u_w,
    mask_u_w_cutmixed1,
    thresh_global,
):
    conf_filter_u_w_without_cutmix = conf_filter_u_w.clone()
    conf_filter_u_w_sample = rearrange(
        conf_filter_u_w_without_cutmix, "n h w -> n 1 h w"
    )
    segments = (corr_map_u_w_cutmixed1 * conf_filter_u_w_sample).bool()
    batch_size, num_segments, _, _ = corr_map_u_w_cutmixed1.shape

    for img_idx in range(batch_size):
        for segment_idx in range(num_segments):
            segment = segments[img_idx, segment_idx]
            segment_ori = corr_map_u_w_cutmixed1[img_idx, segment_idx]
            high_conf_ratio = torch.sum(segment) / torch.sum(segment_ori).clamp(min=1.0)
            if torch.sum(segment) == 0 or high_conf_ratio < thresh_global:
                continue

            unique_cls, count = torch.unique(
                mask_u_w_cutmixed1[img_idx][segment == 1], return_counts=True
            )
            if len(count) == 0:
                continue

            if torch.max(count) / torch.sum(count) > thresh_global:
                top_class = unique_cls[torch.argmax(count)]
                mask_u_w_cutmixed1[img_idx][segment_ori == 1] = top_class
                conf_filter_u_w_without_cutmix[img_idx] = (
                    conf_filter_u_w_without_cutmix[img_idx] | segment_ori
                )

    return conf_filter_u_w_without_cutmix | conf_filter_u_w, mask_u_w_cutmixed1


def masked_mean(loss_map, valid_mask, normalizer):
    return (loss_map * valid_mask).sum() / normalizer


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

    _, backbone_version = get_backbone_info(cfg)
    model = build_model(cfg, backbone_version)
    load_backbone_checkpoint(model, cfg)

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
    log_cuda_memory(
        logger,
        rank,
        "after_ddp_wrap",
        local_rank=local_rank,
        save_path=args.save_path,
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
    previous_best = 0.0
    best_epoch = 0
    epoch = -1

    latest_path = os.path.join(args.save_path, "latest.pth")
    best_path = os.path.join(args.save_path, "best.pth")
    if os.path.exists(latest_path):
        log_cuda_memory(logger, rank, "before_resume_load", save_path=args.save_path)
        checkpoint = load_checkpoint_on_cpu(latest_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        log_cuda_memory(logger, rank, "after_resume_load", save_path=args.save_path)
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        best_epoch = checkpoint["best_epoch"]

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}".format(
                    epoch, previous_best, best_epoch
                )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_x_corr = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_kl = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_loss_corr = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        model.train()

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
            (img_u_w_mix, img_u_s1_mix, _, ignore_mask_mix, _, _),
        ) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, ignore_mask = img_u_s1.cuda(), ignore_mask.cuda()
            cutmix_box1 = cutmix_box1.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix = img_u_s1_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()
                res_u_w_mix = model(img_u_w_mix, need_fp=False, use_corr=False)
                pred_u_w_mix = res_u_w_mix.detach()
                conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)
            model.train()

            cutmix_box1_img = cutmix_box1.unsqueeze(1).bool()
            img_u_s1 = torch.where(cutmix_box1_img, img_u_s1_mix, img_u_s1)

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            res_w = model(torch.cat((img_x, img_u_w)), need_fp=True, use_corr=True)
            preds = res_w["out"]
            preds_fp = res_w["out_fp"]
            preds_corr = res_w["corr_out"]
            preds_corr_map = res_w["corr_map"].detach()

            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_x_corr, pred_u_w_corr = preds_corr.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            pred_u_w_corr_map = preds_corr_map[num_lb:]
            del res_w, preds, preds_fp, preds_corr, preds_corr_map

            res_s = model(img_u_s1, need_fp=False, use_corr=True)
            pred_u_s1 = res_s["out"]
            pred_u_s1_corr = res_s["corr_out"]
            del res_s

            with torch.no_grad():
                pred_u_w = pred_u_w.detach()
                conf_u_w, mask_u_w = pred_u_w.softmax(dim=1).max(dim=1)

            mask_u_w_cutmixed1 = mask_u_w.clone()
            conf_u_w_cutmixed1 = conf_u_w.clone()
            ignore_mask_cutmixed1 = ignore_mask.clone()
            corr_map_u_w_cutmixed1 = pred_u_w_corr_map

            cutmix_box1_map = cutmix_box1 == 1
            mask_u_w_cutmixed1[cutmix_box1_map] = mask_u_w_mix[cutmix_box1_map]
            conf_u_w_cutmixed1[cutmix_box1_map] = conf_u_w_mix[cutmix_box1_map]
            ignore_mask_cutmixed1[cutmix_box1_map] = ignore_mask_mix[cutmix_box1_map]

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

            conf_filter_u_w = (conf_u_w_cutmixed1 >= thresh_global) & (
                ignore_mask_cutmixed1 != 255
            )
            conf_filter_u_w_without_cutmix, mask_u_w_cutmixed1 = (
                refine_corr_pseudo_labels(
                    corr_map_u_w_cutmixed1,
                    conf_filter_u_w,
                    mask_u_w_cutmixed1,
                    thresh_global,
                )
            )

            valid_u = (ignore_mask != 255).sum().clamp(min=1.0)
            valid_u_cutmixed = (ignore_mask_cutmixed1 != 255).sum().clamp(min=1.0)

            loss_x = criterion_l(pred_x, mask_x)
            loss_x_corr = criterion_l(pred_x_corr, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = masked_mean(
                loss_u_s1, conf_filter_u_w_without_cutmix, valid_u_cutmixed
            )

            loss_u_corr_s = criterion_u(pred_u_s1_corr, mask_u_w_cutmixed1)
            loss_u_corr_s = masked_mean(
                loss_u_corr_s, conf_filter_u_w_without_cutmix, valid_u_cutmixed
            )

            loss_u_corr_w = criterion_u(pred_u_w_corr, mask_u_w)
            weak_valid_mask = (conf_u_w >= thresh_global) & (ignore_mask != 255)
            loss_u_corr_w = masked_mean(loss_u_corr_w, weak_valid_mask, valid_u)
            loss_u_corr = 0.5 * (loss_u_corr_s + loss_u_corr_w)

            softmax_pred_u_w = F.softmax(pred_u_w, dim=1)
            logsoftmax_pred_u_s1 = F.log_softmax(pred_u_s1, dim=1)
            loss_u_kl = criterion_kl(logsoftmax_pred_u_s1, softmax_pred_u_w)
            loss_u_kl = masked_mean(
                loss_u_kl.sum(dim=1), conf_filter_u_w, valid_u_cutmixed
            )

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = masked_mean(loss_u_w_fp, weak_valid_mask, valid_u)

            loss = (
                0.5 * loss_x
                + 0.5 * loss_x_corr
                + 0.25 * loss_u_s1
                + 0.25 * loss_u_kl
                + 0.25 * loss_u_w_fp
                + 0.25 * loss_u_corr
            ) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_x_corr.update(loss_x_corr.item())
            total_loss_s.update(loss_u_s1.item())
            total_loss_kl.update(loss_u_kl.item())
            total_loss_w_fp.update(loss_u_w_fp.item())
            total_loss_corr.update(loss_u_corr.item())

            mask_ratio = (
                (conf_u_w >= thresh_global) & (ignore_mask != 255)
            ).sum().item() / valid_u.item()
            total_mask_ratio.update(mask_ratio)

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_x_corr", loss_x_corr.item(), iters)
                writer.add_scalar("train/loss_s", loss_u_s1.item(), iters)
                writer.add_scalar("train/loss_kl", loss_u_kl.item(), iters)
                writer.add_scalar("train/loss_w_fp", loss_u_w_fp.item(), iters)
                writer.add_scalar("train/loss_corr", loss_u_corr.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)
                writer.add_scalar(
                    "train/thresh_global",
                    thresh_global.item()
                    if torch.is_tensor(thresh_global)
                    else float(thresh_global),
                    iters,
                )

            if i % max(len(trainloader_u) // 8, 1) == 0 and rank == 0:
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total: {:.3f}, Loss x: {:.3f}, Loss x_corr: {:.3f}, "
                    "Loss s: {:.3f}, Loss kl: {:.3f}, Loss fp: {:.3f}, Loss corr: {:.3f}, Mask ratio: {:.3f}, Thresh: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_x_corr.avg,
                        total_loss_s.avg,
                        total_loss_kl.avg,
                        total_loss_w_fp.avg,
                        total_loss_corr.avg,
                        total_mask_ratio.avg,
                        thresh_global.item()
                        if torch.is_tensor(thresh_global)
                        else float(thresh_global),
                    )
                )

        mIoU, iou_class = validation_cpu(cfg, model, valloader)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                writer.add_scalar(f"eval/{CLASSES[cfg['dataset']][cls_idx]}", iou, epoch)
            writer.add_scalar("eval/mIoU", mIoU, epoch)

            is_best = mIoU > previous_best
            previous_best = max(previous_best, mIoU)
            if is_best:
                best_epoch = epoch

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
                "best_epoch": best_epoch,
            }
            save_checkpoint_to_disk(
                checkpoint,
                latest_path=latest_path,
                best_path=best_path,
                is_best=is_best,
            )

            logger.info(
                "***** Evaluation ***** >>>> Epoch: {:}, mIoU: {:.2f}, Previous best: {:.2f} @epoch-{}".format(
                    epoch, mIoU, previous_best, best_epoch
                )
            )


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
