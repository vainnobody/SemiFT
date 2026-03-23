import argparse
from copy import deepcopy
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
from model.semseg.dpt import DPT
from model.semseg.my_upernet import MyUperNet
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

from util.viz import Visualizer


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Reproduced FixMatch with an EMA Teacher for Semi-Supervised Semantic Segmentation"
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
        model = DPT(
            **model_kwargs,
            backbone_version=backbone_version,
        )
    elif cfg["model"] == "upernet":
        model = UperNet(
            **model_kwargs,
            backbone_version=backbone_version,
        )

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
    log_cuda_memory(logger, rank, "after_model_to_cuda", local_rank=local_rank, save_path=args.save_path)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )
    log_cuda_memory(logger, rank, "after_ddp_wrap", local_rank=local_rank, save_path=args.save_path)

    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

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
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        log_cuda_memory(logger, rank, "before_resume_load", save_path=args.save_path)
        checkpoint = load_checkpoint_on_cpu(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        model_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        log_cuda_memory(logger, rank, "after_resume_load", save_path=args.save_path)
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        previous_best_ema = checkpoint["previous_best_ema"]
        best_epoch = checkpoint["best_epoch"]
        best_epoch_ema = checkpoint["best_epoch_ema"]

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
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        model.train()

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
        ) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = (
                img_u_w.cuda(),
                img_u_s1.cuda(),
                img_u_s2.cuda(),
            )
            ignore_mask, cutmix_box1, cutmix_box2 = (
                ignore_mask.cuda(),
                cutmix_box1.cuda(),
                cutmix_box2.cuda(),
            )

            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = (
                img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            )
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = (
                img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            )

            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(
                torch.cat((img_u_s1, img_u_s2)), comp_drop=True
            ).chunk(2)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = (
                mask_u_w.clone(),
                conf_u_w.clone(),
                ignore_mask.clone(),
            )
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = (
                mask_u_w.clone(),
                conf_u_w.clone(),
                ignore_mask.clone(),
            )

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[
                cutmix_box1 == 1
            ]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[
                cutmix_box2 == 1
            ]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * (
                (conf_u_w_cutmixed1 >= cfg["conf_thresh"])
                & (ignore_mask_cutmixed1 != 255)
            )
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * (
                (conf_u_w_cutmixed2 >= cfg["conf_thresh"])
                & (ignore_mask_cutmixed2 != 255)
            )
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0

            loss = (loss_x + loss_u_s) / 2.0

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
                        "img_u_s1": (img_u_s1[0], Visualizer.TENSOR),
                        "img_u_s2": (img_u_s2[0], Visualizer.TENSOR),
                        "mask_u_w_cutmixed1": (
                            mask_u_w_cutmixed1[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "mask_u_w_cutmixed2": (
                            mask_u_w_cutmixed2[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_u_s1": (
                            pred_u_s1.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_u_s2": (
                            pred_u_s2.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = (
                (conf_u_w >= cfg["conf_thresh"]) & (ignore_mask != 255)
            ).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

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
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_s", loss_u_s.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: "
                    "{:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_mask_ratio.avg,
                    )
                )

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode", "slide_window" if cfg["dataset"] == "cityscapes" else "original"
        )
        val_cfg.setdefault("ignore_index", cfg.get("ignore_index", 255))
        eval_mode = val_cfg["eval_mode"]
        # mIoU, iou_class = evaluate(
        #     model, valloader, eval_mode, cfg, multiplier=patch_size
        # )
        # mIoU_ema, iou_class_ema = evaluate(
        #     model_ema, valloader, eval_mode, cfg, multiplier=patch_size
        # )

        mIoU, iou_class = validation_cpu(val_cfg, model, valloader)
        mIoU_ema, iou_class_ema = validation_cpu(val_cfg, model_ema, valloader)

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
