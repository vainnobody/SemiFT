import argparse
import logging
import os
import pprint
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
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
    classmix_batch,
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
    maybe_load_checkpoint,
    save_checkpoint,
    update_ema,
    update_lr,
    wrap_ddp,
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
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(trainset_l),
    )
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(trainset_u),
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(valset),
    )
    return trainloader_l, trainloader_u, valloader


def unpack_queue_state(checkpoint, nclass, feat_dim, bank_size):
    payload = checkpoint.get("segmind_queue")
    if payload is None:
        return init_queue_state(nclass, feat_dim, bank_size)
    return load_queue_state(payload)


def main(args, cfg):
    logger, rank, world_size, writer = build_logger_and_runtime(args, cfg)

    if rank == 0:
        logger.info("{}\n".format(pprint.pformat({**cfg, **vars(args), "ngpus": world_size})))

    model = build_model(cfg)
    load_result = load_backbone_checkpoint(model, cfg)
    if cfg.get("lock_backbone"):
        model.lock_backbone()
    log_model_info(logger, rank, model, load_result=load_result)

    optimizer = torch.optim.AdamW(
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
                "lr": cfg["lr"] * cfg.get("lr_multi", 1.0),
            },
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    model, local_rank = wrap_ddp(model, logger=logger, rank=rank, save_path=args.save_path)
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    criterion_l = build_labeled_criterion(cfg, local_rank)
    entropy_criterion = nn.MSELoss().cuda(local_rank)

    trainloader_l, trainloader_u, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best = 0.0
    best_epoch = 0
    start_epoch = -1

    proj_dim = cfg.get("proj_dim", 256)
    queue_state = init_queue_state(cfg["nclass"], proj_dim, cfg.get("bank_size", 10000))

    latest_path = os.path.join(args.save_path, "latest.pth")
    resume_state = maybe_load_checkpoint(
        args,
        model,
        optimizer,
        model_ema=model_ema,
        logger=logger,
        rank=rank,
    )
    if os.path.exists(latest_path):
        checkpoint = load_checkpoint_on_cpu(latest_path)
        previous_best = resume_state.get("previous_best", previous_best)
        best_epoch = resume_state.get("best_epoch", best_epoch)
        start_epoch = resume_state.get("epoch", start_epoch)
        queue_state = unpack_queue_state(
            checkpoint,
            cfg["nclass"],
            proj_dim,
            cfg.get("bank_size", 10000),
        )
        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n", start_epoch)

    alpha_ema = cfg.get("alpha_ema", 0.99)
    epoch_pre = cfg.get("epoch_pre", cfg["epochs"])
    conf_thresh = cfg.get("conf_thresh", 0.7)
    mask_rate = cfg.get("mask_rate", cfg.get("mask_rate_end", 0.25))
    mask_gap = cfg.get("mask_gap", 16)

    for epoch in range(start_epoch + 1, cfg["epochs"]):
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        model.train()

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
                "===========> Epoch: %s, Previous best: %.2f @epoch-%s",
                epoch,
                previous_best,
                best_epoch,
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
                ) = classmix_batch(
                    img_u_w,
                    img_u_s,
                    pseudo_label_u.float(),
                    pseudo_conf_u,
                    teacher_entropy_u,
                    valid_mask_u.float(),
                    labels=pseudo_label_u,
                )
                pseudo_label_mix = pseudo_label_mix.long()
                pseudo_label_mix[valid_mask_mix < 0.5] = cfg["ignore_index"]

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
            lr = update_lr(optimizer, cfg, iters, total_iters)
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
                    "Iters: %s, LR: %.7f, Total: %.3f, L: %.3f, E: %.3f, R: %.3f, RSC: %.3f, C: %.3f, PConf: %.3f, conf_thresh: %.3f",
                    step,
                    optimizer.param_groups[0]["lr"],
                    meters["loss_all"].avg,
                    meters["loss_l"].avg,
                    meters["loss_e"].avg,
                    meters["loss_r"].avg,
                    meters["loss_rsc"].avg,
                    meters["loss_c"].avg,
                    meters["pseudo_conf"].avg,
                    conf_thresh,
                )

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode",
            "slide_window" if cfg["dataset"] == "cityscapes" else "original",
        )
        val_cfg.setdefault("ignore_index", cfg.get("ignore_index", 255))
        mIoU, iou_class = validation_cpu(val_cfg, model, valloader)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [%s %s] IoU: %.2f",
                    cls_idx,
                    CLASSES[cfg["dataset"]][cls_idx],
                    iou,
                )
            logger.info(
                "***** Evaluation %s ***** >>>> MeanIoU: %.2f\n",
                val_cfg["eval_mode"],
                mIoU,
            )
            writer.add_scalar("eval/mIoU", mIoU, epoch)
            for cls_idx, iou in enumerate(iou_class):
                writer.add_scalar(
                    f"eval/{CLASSES[cfg['dataset']][cls_idx]}_IoU",
                    iou,
                    epoch,
                )

        is_best = mIoU >= previous_best
        previous_best = max(mIoU, previous_best)
        if is_best:
            best_epoch = epoch

        save_checkpoint(
            args,
            rank,
            model,
            optimizer,
            epoch,
            previous_best,
            best_epoch,
            model_ema=model_ema,
            extra={"segmind_queue": serialize_queue_state(queue_state)},
            is_best=is_best,
        )


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
