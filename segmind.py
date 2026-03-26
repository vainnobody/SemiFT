import argparse
import logging
import os
import pprint
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

from dataset.semi_segmind import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt_segmind import DPT_SegMind
from model.semseg.upernet_segmind import UPerNet_SegMind
from util.classes import CLASSES
from util.focal import FocalLoss
from util.ohem import ProbOhemCrossEntropy2d
from util.segmind_utils import (
    classmix_batch as segmind_classmix_batch,
    compute_contrastive_loss,
    compute_masked_segmentation_loss,
    compute_reconstruction_loss,
    get_batch_mask_tensor,
    init_queue_state,
    load_queue_state,
    serialize_queue_state,
)
from util.ssl_method_utils import (
    build_logger_and_runtime,
    get_backbone_info,
    get_model_kwargs,
    load_backbone_checkpoint,
    load_checkpoint_on_cpu,
    log_model_info,
    update_ema,
    save_checkpoint_to_disk,
)
from util.utils import AverageMeter
from util.validation import validation_cpu as shared_validation_cpu


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def get_parser():
    parser = argparse.ArgumentParser(
        description="SegMind port on top of the SemiFT training scaffold"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def build_model(cfg):
    model_kwargs = get_model_kwargs(cfg)
    _, backbone_version = get_backbone_info(cfg)
    common_kwargs = {
        **model_kwargs,
        "backbone_version": backbone_version,
        "proj_dim": cfg.get("proj_dim", 256),
        "recon_channels": cfg.get("recon_channels", 128),
    }
    if cfg["model"] == "dpt":
        model = DPT_SegMind(**common_kwargs)
    elif cfg["model"] == "upernet":
        model = UPerNet_SegMind(**common_kwargs)
    else:
        raise ValueError(f"Unsupported SegMind model '{cfg['model']}'.")
    return model


def build_labeled_criterion(cfg, local_rank):
    if cfg["criterion"]["name"] == "CELoss":
        return nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    if cfg["criterion"]["name"] == "OHEM":
        return ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    if cfg["criterion"]["name"] == "FocalLoss":
        return FocalLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    raise NotImplementedError(cfg["criterion"]["name"])


def update_lr_official(optimizer, base_lr, iters, total_iters, power=0.9, min_lr=1e-6):
    lr = max(base_lr * (1 - iters / total_iters) ** power, min_lr)
    optimizer.param_groups[0]["lr"] = lr
    for group_idx in range(1, len(optimizer.param_groups)):
        optimizer.param_groups[group_idx]["lr"] = (
            lr * optimizer.param_groups[group_idx].get("lr_scale", 1.0)
        )
    return lr


def apply_segmind_defaults(cfg):
    cfg = dict(cfg)
    cfg.setdefault("proj_dim", 256)
    cfg.setdefault("recon_channels", 128)
    cfg.setdefault("bank_size", 10000)
    cfg.setdefault("num_query", 256)
    cfg.setdefault("num_negative", 512)
    cfg.setdefault("temperature", 0.5)
    cfg.setdefault("query_threshold", 0.97)
    cfg.setdefault("pseudo_threshold", cfg.get("conf_thresh", 0.95))
    cfg.setdefault("alpha_ema", 0.99)
    cfg.setdefault("epoch_pre", 0)
    cfg.setdefault("mask_gap", 4)
    cfg.setdefault("mask_rate", cfg.get("mask_rate_end", 0.25))
    cfg.setdefault("lambda_l", 1.0)
    cfg.setdefault("lambda_e", 0.1)
    cfg.setdefault("lambda_r", 0.0)
    cfg.setdefault("lambda_rsc", 0.0)
    cfg.setdefault("lambda_c", 0.0)
    return cfg


def validate_segmind_recipe(cfg):
    crop_size = cfg["crop_size"]
    mask_gap = cfg["mask_gap"]
    if isinstance(crop_size, int):
        valid = crop_size % mask_gap == 0
    else:
        valid = all(size % mask_gap == 0 for size in crop_size)
    if not valid:
        raise ValueError(
            f"SegMind requires crop_size divisible by mask_gap, got crop_size={crop_size} mask_gap={mask_gap}."
        )


def build_optimizer(cfg, model):
    lr_multi = cfg.get("lr_multi", 1.0)
    return AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": cfg["lr"],
                "lr_scale": 1.0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "backbone" not in name and param.requires_grad
                ],
                "lr": cfg["lr"] * lr_multi,
                "lr_scale": lr_multi,
            },
        ],
        lr=cfg["lr"],
        betas=tuple(cfg.get("betas", (0.9, 0.99))),
        weight_decay=cfg.get("weight_decay", cfg.get("weight_delay", 1e-6)),
    )


def build_dataloaders(args, cfg):
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

    workers = cfg.get("workers", 4)
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
        sampler=trainsampler_l,
    )
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
        sampler=trainsampler_u,
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        drop_last=False,
        sampler=valsampler,
    )
    return trainloader_l, trainloader_u, valloader


def unpack_queue_state(checkpoint, nclass, feat_dim, bank_size):
    payload = checkpoint.get("segmind_queue")
    if payload is None:
        return init_queue_state(nclass, feat_dim, bank_size)
    return load_queue_state(payload)


def create_block_mask(batch_size, height, width, mask_patch=16, mask_ratio=0.25, device=None):
    return get_batch_mask_tensor(
        (batch_size, 1, height, width),
        mask_gap=mask_patch,
        mask_rate=mask_ratio,
        device=device,
    )


def classmix_batch(img_w, img_s, pseudo_label, pseudo_conf, pseudo_entropy, valid_mask):
    mixed = segmind_classmix_batch(
        img_w,
        img_s,
        pseudo_label.float(),
        pseudo_conf,
        pseudo_entropy,
        valid_mask.float(),
        labels=pseudo_label,
    )
    img_w_mix, img_s_mix, pseudo_mix, conf_mix, entropy_mix, valid_mix, _ = mixed
    return img_w_mix, img_s_mix, pseudo_mix.long(), conf_mix, entropy_mix, valid_mix.bool()


def main(args, cfg):
    cfg = apply_segmind_defaults(cfg)
    validate_segmind_recipe(cfg)
    logger, rank, world_size, writer = build_logger_and_runtime(args, cfg)

    if rank == 0:
        logger.info("{}\n".format(pprint.pformat({**cfg, **vars(args), "ngpus": world_size})))

    model = build_model(cfg)
    load_result = load_backbone_checkpoint(model, cfg)
    if cfg.get("lock_backbone"):
        model.lock_backbone()
    log_model_info(logger, rank, model, load_result=load_result)

    optimizer = build_optimizer(cfg, model)

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )
    if hasattr(model, "_set_static_graph"):
        model._set_static_graph()

    model_ema = deepcopy(model)
    model_ema.train()
    for param in model_ema.parameters():
        param.requires_grad = False

    criterion_l = build_labeled_criterion(cfg, local_rank)
    entropy_criterion = nn.MSELoss().cuda(local_rank)

    trainloader_l, trainloader_u, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best = 0.0
    previous_best_ema = 0.0
    best_epoch = 0
    best_epoch_ema = 0
    start_epoch = -1

    proj_dim = cfg["proj_dim"]
    queue_state = init_queue_state(cfg["nclass"], proj_dim, cfg["bank_size"])

    latest_path = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(latest_path):
        checkpoint = load_checkpoint_on_cpu(latest_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "model_ema" in checkpoint:
            model_ema.load_state_dict(checkpoint["model_ema"])
        previous_best = checkpoint.get("previous_best", previous_best)
        previous_best_ema = checkpoint.get("previous_best_ema", previous_best_ema)
        best_epoch = checkpoint.get("best_epoch", best_epoch)
        best_epoch_ema = checkpoint.get("best_epoch_ema", best_epoch_ema)
        start_epoch = checkpoint.get("epoch", start_epoch)
        queue_state = unpack_queue_state(
            checkpoint,
            cfg["nclass"],
            proj_dim,
            cfg["bank_size"],
        )
        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n", start_epoch)

    alpha_ema = cfg["alpha_ema"]
    epoch_pre = cfg["epoch_pre"]
    mask_rate = cfg["mask_rate"]
    mask_gap = cfg["mask_gap"]

    for epoch in range(start_epoch + 1, cfg["epochs"]):
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        model.train()
        model_ema.train()

        meters = {
            "loss_all": AverageMeter(),
            "loss_l": AverageMeter(),
            "loss_e": AverageMeter(),
            "loss_r": AverageMeter(),
            "loss_rsc": AverageMeter(),
            "loss_c": AverageMeter(),
            "pseudo_conf": AverageMeter(),
        }

        if rank == 0:
            logger.info(
                "===========> Epoch: %s, Previous best: %.2f @epoch-%s, EMA: %.2f @epoch-%s",
                epoch,
                previous_best,
                best_epoch,
                previous_best_ema,
                best_epoch_ema,
            )

        for step, ((img_l_w, img_l_s, mask_l), (img_u_w, img_u_s, valid_mask_u)) in enumerate(
            zip(trainloader_l, trainloader_u)
        ):
            img_l_w = img_l_w.cuda(local_rank)
            img_l_s = img_l_s.cuda(local_rank)
            mask_l = mask_l.cuda(local_rank)
            img_u_w = img_u_w.cuda(local_rank)
            img_u_s = img_u_s.cuda(local_rank)
            valid_mask_u = valid_mask_u.cuda(local_rank)

            with torch.no_grad():
                teacher_inputs = torch.cat((img_l_w, img_u_w), dim=0)
                teacher_logits = model_ema(teacher_inputs, return_proj=False)["out"].detach()
                teacher_probs = torch.softmax(teacher_logits, dim=1)
                teacher_entropy = torch.sum(
                    -teacher_probs * torch.log(teacher_probs.clamp_min(1e-8)),
                    dim=1,
                )
                teacher_entropy_l = teacher_entropy[: img_l_w.shape[0]]
                teacher_entropy_u = teacher_entropy[img_l_w.shape[0] :]
                pseudo_probs_u = teacher_probs[img_l_w.shape[0] :]
                pseudo_conf_u, pseudo_label_u = pseudo_probs_u.max(dim=1)

                (
                    img_u_w_mix,
                    img_u_s_mix,
                    pseudo_label_mix,
                    pseudo_conf_mix,
                    entropy_u_mix,
                    valid_mask_mix,
                    _,
                ) = segmind_classmix_batch(
                    img_u_w,
                    img_u_s,
                    pseudo_label_u.float(),
                    pseudo_conf_u,
                    teacher_entropy_u,
                    valid_mask_u.float(),
                    labels=pseudo_label_u,
                )
                pseudo_label_mix = pseudo_label_mix.long()
                pseudo_valid_mask = (valid_mask_mix >= 0.5) & (
                    pseudo_conf_mix >= cfg["pseudo_threshold"]
                )
                pseudo_label_mix[~pseudo_valid_mask] = cfg["ignore_index"]

            strong_inputs = torch.cat((img_l_s, img_u_s_mix), dim=0)
            student_outputs = model(
                strong_inputs,
                return_proj=cfg.get("lambda_c", 1.0) != 0,
            )
            student_logits = student_outputs["out"]
            student_probs = torch.softmax(student_logits, dim=1)
            student_entropy = torch.sum(
                -student_probs * torch.log(student_probs.clamp_min(1e-8)),
                dim=1,
            )

            labels_all = torch.cat((mask_l, pseudo_label_mix), dim=0)
            loss_l = criterion_l(student_logits, labels_all)
            entropy_targets = torch.cat((teacher_entropy_l, entropy_u_mix), dim=0)
            loss_e = entropy_criterion(student_entropy, entropy_targets)

            loss_r = student_logits.new_zeros(())
            loss_rsc = student_logits.new_zeros(())
            if (cfg.get("lambda_r", 1.0) != 0 or cfg.get("lambda_rsc", 1.0) != 0) and epoch <= epoch_pre:
                weak_inputs = torch.cat((img_l_w, img_u_w_mix), dim=0)
                mask_tensor = get_batch_mask_tensor(
                    weak_inputs.shape,
                    mask_gap=mask_gap,
                    mask_rate=mask_rate,
                    device=weak_inputs.device,
                )
                masked_inputs = weak_inputs * mask_tensor
                recon_outputs = model(
                    masked_inputs,
                    return_proj=False,
                    return_reconstruction=True,
                    reconstruction_mask=mask_tensor,
                )
                if cfg.get("lambda_r", 1.0) != 0:
                    loss_r = compute_reconstruction_loss(
                        recon_outputs["recon"],
                        weak_inputs,
                        mask_tensor.squeeze(1),
                    )
                if cfg.get("lambda_rsc", 1.0) != 0:
                    loss_rsc = compute_masked_segmentation_loss(
                        recon_outputs["out"],
                        labels_all,
                        mask_tensor.squeeze(1),
                        cfg["ignore_index"],
                    )

            loss_c = student_logits.new_zeros(())
            if cfg.get("lambda_c", 1.0) != 0:
                loss_c = compute_contrastive_loss(
                    student_outputs["proj_feat"],
                    labels_all,
                    student_probs,
                    queue_state,
                    query_threshold=cfg.get("query_threshold", 0.97),
                    temperature=cfg.get("temperature", 0.5),
                    num_query=cfg.get("num_query", 256),
                    num_negative=cfg.get("num_negative", 512),
                    ignore_index=cfg["ignore_index"],
                )

            loss = (
                cfg.get("lambda_l", 1.0) * loss_l
                + cfg.get("lambda_e", 1.0) * loss_e
                + cfg.get("lambda_r", 1.0) * loss_r
                + cfg.get("lambda_rsc", 1.0) * loss_rsc
                + cfg.get("lambda_c", 1.0) * loss_c
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader_u) + step
            lr = update_lr_official(
                optimizer,
                base_lr=cfg["lr"],
                iters=iters,
                total_iters=total_iters,
                power=cfg.get("lr_power", 0.9),
                min_lr=cfg.get("min_lr", 1e-6),
            )
            update_ema(model, model_ema, iters, max_decay=alpha_ema)

            meters["loss_all"].update(loss.item())
            meters["loss_l"].update(loss_l.item())
            meters["loss_e"].update(loss_e.item())
            meters["loss_r"].update(loss_r.item())
            meters["loss_rsc"].update(loss_rsc.item())
            meters["loss_c"].update(loss_c.item())
            meters["pseudo_conf"].update(pseudo_conf_mix.mean().item())

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_l", loss_l.item(), iters)
                writer.add_scalar("train/loss_e", loss_e.item(), iters)
                writer.add_scalar("train/loss_r", loss_r.item(), iters)
                writer.add_scalar("train/loss_rsc", loss_rsc.item(), iters)
                writer.add_scalar("train/loss_c", loss_c.item(), iters)
                writer.add_scalar("train/pseudo_conf", pseudo_conf_mix.mean().item(), iters)
                writer.add_scalar("train/lr", lr, iters)

            if rank == 0 and step % max(1, len(trainloader_u) // 8) == 0:
                logger.info(
                    "Iters: %s, LR: %.7f, Total: %.3f, L: %.3f, E: %.3f, R: %.3f, RSC: %.3f, C: %.3f, PConf: %.3f",
                    step,
                    optimizer.param_groups[0]["lr"],
                    meters["loss_all"].avg,
                    meters["loss_l"].avg,
                    meters["loss_e"].avg,
                    meters["loss_r"].avg,
                    meters["loss_rsc"].avg,
                    meters["loss_c"].avg,
                    meters["pseudo_conf"].avg,
                )

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode",
            "slide_window" if cfg["dataset"] == "cityscapes" else "original",
        )
        val_cfg.setdefault("ignore_index", cfg.get("ignore_index", 255))
        mIoU, iou_class = validation_cpu(val_cfg, model, valloader)
        mIoU_ema, iou_class_ema = validation_cpu(val_cfg, model_ema, valloader)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [%s %s] IoU: %.2f, EMA: %.2f",
                    cls_idx,
                    CLASSES[cfg["dataset"]][cls_idx],
                    iou,
                    iou_class_ema[cls_idx],
                )
            logger.info(
                "***** Evaluation %s ***** >>>> MeanIoU: %.2f, EMA: %.2f\n",
                val_cfg["eval_mode"],
                mIoU,
                mIoU_ema,
            )
            writer.add_scalar("eval/mIoU", mIoU, epoch)
            writer.add_scalar("eval/mIoU_ema", mIoU_ema, epoch)
            for cls_idx, iou in enumerate(iou_class):
                writer.add_scalar(f"eval/{CLASSES[cfg['dataset']][cls_idx]}_IoU", iou, epoch)
                writer.add_scalar(f"eval/{CLASSES[cfg['dataset']][cls_idx]}_IoU_ema", iou_class_ema[cls_idx], epoch)

        is_best = mIoU >= previous_best
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if is_best:
            best_epoch = epoch
        if mIoU_ema >= previous_best_ema:
            best_epoch_ema = epoch

        checkpoint = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "previous_best": previous_best,
            "previous_best_ema": previous_best_ema,
            "best_epoch": best_epoch,
            "best_epoch_ema": best_epoch_ema,
            "segmind_queue": serialize_queue_state(queue_state),
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
