import argparse
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

from dataset.semi_rs import SemiDataset
from dataset.val import ValDataset
from model.semseg.corrmatch import DPT_CorrMatch, UPerNet_CorrMatch
from util.classes import CLASSES
from util.corrmatch_utils import ThreshController, apply_region_propagation
from util.focal import FocalLoss
from util.ohem import ProbOhemCrossEntropy2d
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


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def get_parser():
    parser = argparse.ArgumentParser(
        description="CorrMatch for Semi-Supervised Semantic Segmentation"
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

    model_kwargs = get_model_kwargs(cfg)
    _, backbone_version = get_backbone_info(cfg)

    if cfg["model"] == "dpt":
        model = DPT_CorrMatch(
            **model_kwargs,
            backbone_version=backbone_version,
        )
    elif cfg["model"] == "upernet":
        model = UPerNet_CorrMatch(
            **model_kwargs,
            backbone_version=backbone_version,
        )
    else:
        raise ValueError(f'Unsupported model {cfg["model"]!r} for CorrMatch.')

    load_backbone_checkpoint(model, cfg)

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
        find_unused_parameters=False,
    )
    log_cuda_memory(
        logger,
        rank,
        "after_ddp_wrap",
        local_rank=local_rank,
        save_path=args.save_path,
    )

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

    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)
    criterion_kl = nn.KLDivLoss(reduction="none").cuda(local_rank)
    thresh_controller = ThreshController(
        nclass=cfg["nclass"],
        momentum=cfg.get("thresh_momentum", 0.999),
        thresh_init=cfg.get("thresh_init", 0.85),
    )

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
    previous_best = 0.0
    best_epoch = 0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        log_cuda_memory(logger, rank, "before_resume_load", save_path=args.save_path)
        checkpoint = load_checkpoint_on_cpu(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        best_epoch = checkpoint.get("best_epoch", 0)
        thresh_value = checkpoint.get("thresh_global")
        if thresh_value is not None:
            thresh_controller.thresh_global = thresh_value.cuda(local_rank)
        log_cuda_memory(logger, rank, "after_resume_load", save_path=args.save_path)
        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, Thresh: {:.4f}".format(
                    epoch,
                    previous_best,
                    best_epoch,
                    thresh_controller.get_thresh_global().item(),
                )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_corr_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_corr_u = AverageMeter()
        total_loss_kl = AverageMeter()
        total_loss_fp = AverageMeter()
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
                pred_u_w_mix = res_u_w_mix["out"].detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                img_u_s1[cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1] = (
                    img_u_s1_mix[cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1]
                )
                model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            res_w = model(torch.cat((img_x, img_u_w)), need_fp=True, use_corr=True)
            preds = res_w["out"]
            preds_fp = res_w["out_fp"]
            preds_corr = res_w["corr_out"]
            preds_corr_map = res_w["corr_map"].detach()
            pred_x_corr, pred_u_w_corr = preds_corr.split([num_lb, num_ulb])
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            pred_u_w_corr_map = preds_corr_map[num_lb:]

            res_s = model(img_u_s1, need_fp=False, use_corr=True)
            pred_u_s1 = res_s["out"]
            pred_u_s1_corr = res_s["corr_out"]

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1 = mask_u_w.clone()
            conf_u_w_cutmixed1 = conf_u_w.clone()
            ignore_mask_cutmixed1 = ignore_mask.clone()

            cutmix_region = cutmix_box1 == 1
            mask_u_w_cutmixed1[cutmix_region] = mask_u_w_mix[cutmix_region]
            conf_u_w_cutmixed1[cutmix_region] = conf_u_w_mix[cutmix_region]
            ignore_mask_cutmixed1[cutmix_region] = ignore_mask_mix[cutmix_region]

            corr_map_u_w_cutmixed1 = pred_u_w_corr_map.clone()
            corr_map_u_w_cutmixed1 = corr_map_u_w_cutmixed1 & (
                ~cutmix_region.unsqueeze(1)
            ) & (ignore_mask_cutmixed1 != ignore_index).unsqueeze(1)

            thresh_controller.thresh_update(
                pred_u_w.detach(), ignore_mask_cutmixed1, update_g=True
            )
            thresh_global = thresh_controller.get_thresh_global()

            conf_filter_u_w = (conf_u_w_cutmixed1 >= thresh_global) & (
                ignore_mask_cutmixed1 != ignore_index
            )
            mask_u_w_cutmixed1, conf_filter_u_w = apply_region_propagation(
                mask_u_w_cutmixed1,
                corr_map_u_w_cutmixed1,
                conf_filter_u_w,
                thresh_global,
            )

            loss_x = criterion_l(pred_x, mask_x)
            loss_x_corr = criterion_l(pred_x_corr, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = (loss_u_s1 * conf_filter_u_w).sum() / (
                (ignore_mask_cutmixed1 != ignore_index).sum().clamp(min=1.0)
            )

            loss_u_corr_s1 = criterion_u(pred_u_s1_corr, mask_u_w_cutmixed1)
            loss_u_corr_s1 = (loss_u_corr_s1 * conf_filter_u_w).sum() / (
                (ignore_mask_cutmixed1 != ignore_index).sum().clamp(min=1.0)
            )
            weak_corr_mask = (conf_u_w >= thresh_global) & (ignore_mask != ignore_index)
            loss_u_corr_w = criterion_u(pred_u_w_corr, mask_u_w)
            loss_u_corr_w = (loss_u_corr_w * weak_corr_mask).sum() / (
                (ignore_mask != ignore_index).sum().clamp(min=1.0)
            )
            loss_u_corr = 0.5 * (loss_u_corr_s1 + loss_u_corr_w)

            softmax_pred_u_w = pred_u_w.softmax(dim=1)
            logsoftmax_pred_u_s1 = pred_u_s1.log_softmax(dim=1)
            loss_u_kl = criterion_kl(logsoftmax_pred_u_s1, softmax_pred_u_w)
            loss_u_kl = (loss_u_kl.sum(dim=1) * conf_filter_u_w).sum() / (
                (ignore_mask_cutmixed1 != ignore_index).sum().clamp(min=1.0)
            )

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = (loss_u_w_fp * weak_corr_mask).sum() / (
                (ignore_mask != ignore_index).sum().clamp(min=1.0)
            )

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
            total_loss_corr_x.update(loss_x_corr.item())
            total_loss_s.update(loss_u_s1.item())
            total_loss_corr_u.update(loss_u_corr.item())
            total_loss_kl.update(loss_u_kl.item())
            total_loss_fp.update(loss_u_w_fp.item())
            mask_ratio = weak_corr_mask.sum().float() / (
                (ignore_mask != ignore_index).sum().float().clamp(min=1.0)
            )
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_x_corr", loss_x_corr.item(), iters)
                writer.add_scalar("train/loss_s", loss_u_s1.item(), iters)
                writer.add_scalar("train/loss_corr_u", loss_u_corr.item(), iters)
                writer.add_scalar("train/loss_kl", loss_u_kl.item(), iters)
                writer.add_scalar("train/loss_fp", loss_u_w_fp.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio.item(), iters)
                writer.add_scalar(
                    "train/thresh_global", thresh_global.item(), iters
                )

            if (i % max(len(trainloader_u) // 8, 1) == 0) and rank == 0:
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total: {:.3f}, Lx: {:.3f}, Lx_corr: {:.3f}, "
                    "Ls: {:.3f}, Lcorr_u: {:.3f}, Lkl: {:.3f}, Lfp: {:.3f}, Mask: {:.3f}, Thresh: {:.4f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_corr_x.avg,
                        total_loss_s.avg,
                        total_loss_corr_u.avg,
                        total_loss_kl.avg,
                        total_loss_fp.avg,
                        total_mask_ratio.avg,
                        thresh_global.item(),
                    )
                )

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode", "slide_window" if cfg["dataset"] == "cityscapes" else "original"
        )
        val_cfg.setdefault("ignore_index", ignore_index)
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
            logger.info("***** Evaluation ***** >>>> MeanIoU: {:.2f}\n".format(mIoU))
            writer.add_scalar("eval/mIoU", mIoU, epoch)

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
                "thresh_global": thresh_controller.get_thresh_global().detach().cpu(),
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
    main(args, cfg)
