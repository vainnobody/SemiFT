import argparse
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

from dataset.semi_dwl import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt_dwl import DPT_DWL
from model.semseg.upernet_dwl import UPerNet_DWL
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
from util.dwl_utils import (
    init_cls_memory,
    update_cls_memory,
    sample_cls_bins,
    calc_wgt_bins,
    downsample_for_memory,
    move_cls_memory_to_device,
)


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def transfer_pseudo_head(model, alpha):
    if alpha <= 0:
        return

    module = model.module if hasattr(model, "module") else model

    if hasattr(module, "pseudo_head"):
        target_module = module.head
        source_module = module.pseudo_head
    elif hasattr(module, "pseudo_classifier"):
        target_module = module.decoder.classifier
        source_module = module.pseudo_classifier
    else:
        raise AttributeError("DWL model is missing a pseudo prediction head")

    for target_param, source_param in zip(
        target_module.parameters(), source_module.parameters()
    ):
        target_param.data.mul_(1 - alpha).add_(source_param.data, alpha=alpha)


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

    model_kwargs = get_model_kwargs(cfg)
    _, backbone_version = get_backbone_info(cfg)

    if cfg["model"] == "dpt":
        model = DPT_DWL(
            **model_kwargs,
            backbone_version=backbone_version,
        )
    elif cfg["model"] == "upernet":
        model = UPerNet_DWL(
            **model_kwargs,
            backbone_version=backbone_version,
        )
    else:
        raise ValueError(f"Unsupported DWL model: {cfg['model']}")

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

    dwl_cfg = cfg.get("dwl", {})
    memory_n_batches = int(dwl_cfg.get("memory_n_batches", 50))
    memory_downsample_size = int(dwl_cfg.get("memory_downsample_size", 64))
    alpha_head_transfer = float(dwl_cfg.get("alpha", 1.0))

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best = 0.0
    best_epoch = 0
    epoch = -1

    cls_memory_u = init_cls_memory(
        cfg["nclass"], device=torch.device("cuda", local_rank)
    )

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        log_cuda_memory(logger, rank, "before_resume_load", save_path=args.save_path)
        checkpoint = load_checkpoint_on_cpu(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        log_cuda_memory(logger, rank, "after_resume_load", save_path=args.save_path)
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        best_epoch = checkpoint["best_epoch"]
        if "cls_memory_u" in checkpoint:
            cls_memory_u = move_cls_memory_to_device(
                checkpoint["cls_memory_u"], torch.device("cuda", local_rank)
            )

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
        total_loss_s = AverageMeter()
        total_wgt_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        labeled_loader = infinite_loader(trainloader_l)
        log_interval = max(1, len(trainloader_u) // 8)

        model.train()

        for i, (img_u_w, img_u_s, valid_mask, _) in enumerate(trainloader_u):
            img_x, mask_x = next(labeled_loader)

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            valid_mask = valid_mask.cuda().bool()

            with torch.no_grad():
                pred_u_w = model(img_u_w).detach()
                prob_u_w = pred_u_w.softmax(dim=1)
                conf_u_w, mask_u_w = prob_u_w.max(dim=1)

            iters = epoch * len(trainloader_u) + i
            prob_uw_bar, pl_uw_bar = downsample_for_memory(
                prob_u_w, mask_u_w, target_size=memory_downsample_size
            )
            cls_memory_u = update_cls_memory(
                cls_memory_u, prob_uw_bar.detach(), pl_uw_bar, memory_n_batches
            )
            cls_bins_u = sample_cls_bins(cls_memory_u)

            conf_u_w_flat = (
                F.interpolate(
                    conf_u_w.unsqueeze(1),
                    size=(memory_downsample_size, memory_downsample_size),
                    mode="nearest",
                )
                .squeeze(1)
                .reshape(-1)
            )
            wgt_u_flat = calc_wgt_bins(
                cls_bins_u, conf_u_w_flat, pl_uw_bar, iters, total_iters
            )

            batch_size = img_u_w.shape[0]
            wgt_u = F.interpolate(
                wgt_u_flat.reshape(
                    batch_size, memory_downsample_size, memory_downsample_size
                ).unsqueeze(1),
                size=img_u_w.shape[2:],
                mode="nearest",
            ).squeeze(1)

            logit_l = model(img_x)
            _, pseudo_pred_u_s = model(img_u_s, return_pseudo_pred=True)

            loss_x = criterion_l(logit_l, mask_x)
            loss_u_map = criterion_u(pseudo_pred_u_s, mask_u_w)
            loss_u_s = (loss_u_map * wgt_u * valid_mask.float()).sum() / valid_mask.sum().clamp(
                min=1.0
            )
            if iters <= memory_n_batches:
                loss_u_s = loss_u_s * 0.0

            loss = loss_x + loss_u_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            wgt_avg = wgt_u[valid_mask].mean().item() if valid_mask.any() else 0.0
            total_wgt_ratio.update(wgt_avg)

            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_s", loss_u_s.item(), iters)
                writer.add_scalar("train/wgt_avg", wgt_avg, iters)

            if i % log_interval == 0 and rank == 0:
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Wgt avg: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_wgt_ratio.avg,
                    )
                )

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode", "slide_window" if cfg["dataset"] == "cityscapes" else "original"
        )
        val_cfg.setdefault("ignore_index", cfg.get("ignore_index", 255))
        eval_mode = val_cfg["eval_mode"]

        mIoU, iou_class = validation_cpu(val_cfg, model, valloader)

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
                    "eval/%s_IoU" % (CLASSES[cfg["dataset"]][i]), iou, epoch
                )

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if mIoU == previous_best:
            best_epoch = epoch

        transfer_pseudo_head(model, alpha_head_transfer)

        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
                "best_epoch": best_epoch,
                "cls_memory_u": cls_memory_u,
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
