import argparse
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import yaml

from dataset.semi_segmind import SegMindDataset
from dataset.val import ValDataset
from model.semseg.segmind import SegMindModel
from util.classes import CLASSES
from util.ssl_method_utils import (
    build_criterions,
    build_ema_model,
    build_logger_and_runtime,
    build_model,
    build_optimizer,
    log_model_info,
    maybe_load_checkpoint,
    save_checkpoint,
    update_ema,
    update_lr,
    wrap_ddp,
)
from util.train_utils import DictAverageMeter
from util.validation import validation_cpu as shared_validation_cpu
from util.viz import Visualizer
from util.segmind_utils import (
    class_mix_batch,
    create_memory_bank,
    gather_pseudo_from_teacher,
    generate_grid_mask,
    percentile_entropy_mask,
    segmind_contrastive_loss,
)


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def apply_ignore_mask_to_labels(labels, ignore_mask, ignore_index):
    if ignore_mask is None:
        return labels
    return labels.masked_fill(ignore_mask == 255, ignore_index)


def build_entropy_targets(mask_l, mixed_entropy, student_entropy, ignore_mask, ignore_index):
    teacher_entropy_all = torch.cat(
        (
            torch.zeros_like(mask_l, dtype=student_entropy.dtype),
            mixed_entropy,
        ),
        dim=0,
    )
    valid_entropy = torch.cat(
        (
            mask_l != ignore_index,
            ignore_mask != 255,
        ),
        dim=0,
    )
    return teacher_entropy_all, valid_entropy


def get_parser():
    parser = argparse.ArgumentParser(
        description="SegMind training adapted to the SemiFT training scaffold"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def build_dataloaders(args, cfg):
    trainset_u = SegMindDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        ignore_index=cfg["ignore_index"],
    )
    trainset_l = SegMindDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(trainset_u.ids),
        ignore_index=cfg["ignore_index"],
    )
    valset = ValDataset(cfg["dataset"], cfg["data_root"], "val", ignore_value=cfg["ignore_index"])

    workers = cfg.get("workers", 4)
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)

    trainloader_l = DataLoader(trainset_l, batch_size=cfg["batch_size"], pin_memory=True, num_workers=workers, drop_last=True, sampler=trainsampler_l)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg["batch_size"], pin_memory=True, num_workers=workers, drop_last=True, sampler=trainsampler_u)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)
    return trainloader_l, trainloader_u, valloader


def main(args, cfg):
    logger, rank, _, writer = build_logger_and_runtime(args, cfg)
    segmind_cfg = cfg.setdefault("segmind", {})
    base_model, load_result = build_model(cfg, method="segmind")
    model = SegMindModel(
        base_model=base_model,
        nclass=cfg["nclass"],
        project_dim=segmind_cfg.get("project_dim", 256),
    )
    optimizer = build_optimizer(cfg, model)
    log_model_info(logger, rank, model, load_result=load_result)
    model, local_rank = wrap_ddp(model, logger=logger, rank=rank, save_path=args.save_path)
    model_ema = build_ema_model(model)
    criterion_l, _ = build_criterions(cfg, local_rank)
    criterion_r = nn.MSELoss().cuda(local_rank)
    criterion_rsc = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"]).cuda(local_rank)

    trainloader_l, trainloader_u, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader_u) * cfg["epochs"]
    state = maybe_load_checkpoint(args, model, optimizer, model_ema=model_ema, logger=logger, rank=rank)
    previous_best = state["previous_best"]
    best_epoch = state["best_epoch"]
    previous_best_ema = state["previous_best_ema"]
    best_epoch_ema = state["best_epoch_ema"]
    start_epoch = state["epoch"] + 1

    device = torch.device("cuda", local_rank)
    memory_bank = create_memory_bank(
        num_classes=cfg["nclass"],
        proj_dim=segmind_cfg.get("project_dim", 256),
        bank_size=segmind_cfg.get("bank_size", 10000),
        device=device,
    )
    latest_extra = None
    if start_epoch > 0:
        checkpoint = torch.load(f"{args.save_path}/latest.pth", map_location="cpu", weights_only=False)
        latest_extra = checkpoint.get("segmind_bank")
    if latest_extra is not None:
        memory_bank.load_state_dict(latest_extra, device)

    from datetime import datetime

    viz = Visualizer(save_dir=f"./viz/{datetime.now().strftime('%Y%m%d_%H%M%S')}", dataset=cfg["dataset"])

    ema_decay = segmind_cfg.get("ema_decay", 0.99)
    query_threshold = segmind_cfg.get("query_threshold", 0.97)
    temperature = segmind_cfg.get("temperature", 0.5)
    num_query = segmind_cfg.get("num_query", 256)
    num_negative = segmind_cfg.get("num_negative", 512)
    mask_gap = segmind_cfg.get("mim_mask_gap", 16)
    mask_rate = segmind_cfg.get("mim_mask_rate", 0.25)
    recon_warmup_epochs = segmind_cfg.get("recon_warmup_epochs", max(cfg["epochs"] // 2, 1))
    entropy_percent = segmind_cfg.get("entropy_percent", 100)
    lambda_l = segmind_cfg.get("lambda_l", 1.0)
    lambda_e = segmind_cfg.get("lambda_e", 1.0)
    lambda_r = segmind_cfg.get("lambda_r", 1.0)
    lambda_rsc = segmind_cfg.get("lambda_rsc", 1.0)
    lambda_c = segmind_cfg.get("lambda_c", 1.0)

    for epoch in range(start_epoch, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: %d, Previous best: %.2f @epoch-%d, EMA: %.2f @epoch-%d",
                epoch,
                previous_best,
                best_epoch,
                previous_best_ema,
                best_epoch_ema,
            )

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        model.train()
        meter = DictAverageMeter()

        for i, ((img_l_w, img_l_s, mask_l), (img_u_w, img_u_s, ignore_u)) in enumerate(zip(trainloader_l, trainloader_u)):
            img_l_w = img_l_w.cuda(local_rank)
            img_l_s = img_l_s.cuda(local_rank)
            mask_l = mask_l.cuda(local_rank)
            img_u_w = img_u_w.cuda(local_rank)
            img_u_s = img_u_s.cuda(local_rank)
            ignore_u = ignore_u.cuda(local_rank)

            _, pseudo_logit, pseudo_label, teacher_entropy = gather_pseudo_from_teacher(model_ema, img_l_w, img_u_w)
            mixed_u = class_mix_batch(
                img_u_w=img_u_w,
                img_s=img_u_s,
                pseudo_label=pseudo_label,
                pseudo_logit=pseudo_logit,
                entropy=teacher_entropy,
                ignore_mask=ignore_u,
                ignore_index=cfg["ignore_index"],
            )

            strong_inputs = torch.cat((img_l_s, mixed_u["img_s"]), dim=0)
            weak_inputs = torch.cat((img_l_w, mixed_u["img_w"]), dim=0)
            pseudo_label = apply_ignore_mask_to_labels(
                mixed_u["pseudo_label"],
                mixed_u["ignore_mask"],
                cfg["ignore_index"],
            )
            label_all = torch.cat((mask_l, pseudo_label), dim=0)

            strong_outputs = model(strong_inputs, return_aux=True)
            seg_logits = strong_outputs["seg_logits"]
            prob_all = seg_logits.softmax(dim=1)
            loss_l = criterion_l(seg_logits, label_all)

            student_entropy = torch.sum(-prob_all * torch.log(prob_all.clamp_min(1e-8)), dim=1)
            teacher_entropy_all, valid_entropy = build_entropy_targets(
                mask_l,
                mixed_u["entropy"],
                student_entropy,
                mixed_u["ignore_mask"],
                cfg["ignore_index"],
            )
            if entropy_percent < 100:
                entropy_mask = percentile_entropy_mask(student_entropy, valid_entropy, entropy_percent)
            else:
                entropy_mask = valid_entropy
            if entropy_mask.any():
                loss_e = F.mse_loss(student_entropy[entropy_mask], teacher_entropy_all[entropy_mask])
            else:
                loss_e = student_entropy.sum() * 0.0

            loss_c = segmind_contrastive_loss(
                feat=strong_outputs["proj_feat"],
                labels=label_all,
                prob=prob_all,
                bank=memory_bank,
                query_threshold=query_threshold,
                temperature=temperature,
                num_queries=num_query,
                num_negative=num_negative,
                ignore_index=cfg["ignore_index"],
            )

            if epoch < recon_warmup_epochs:
                mim_mask = generate_grid_mask(
                    batch=weak_inputs.shape[0],
                    height=weak_inputs.shape[-2],
                    width=weak_inputs.shape[-1],
                    mask_gap=mask_gap,
                    mask_rate=mask_rate,
                    device=weak_inputs.device,
                )
                masked_weak_inputs = weak_inputs * mim_mask
                recon_outputs = model(masked_weak_inputs, mim_mask=mim_mask, return_aux=True)
                recon_target = F.interpolate(weak_inputs, size=recon_outputs["recon_img"].shape[-2:], mode="bilinear", align_corners=False)
                low_mask = F.interpolate(mim_mask, size=recon_outputs["recon_img"].shape[-2:], mode="nearest").bool().expand_as(recon_outputs["recon_img"])
                hidden_region = ~low_mask
                if hidden_region.any():
                    loss_r = criterion_r(recon_outputs["recon_img"][hidden_region], recon_target[hidden_region])
                else:
                    loss_r = recon_outputs["recon_img"].sum() * 0.0
                small_labels = F.interpolate(label_all.float().unsqueeze(1), size=recon_outputs["mask_logits"].shape[-2:], mode="nearest").squeeze(1).long()
                small_mask = (~F.interpolate(mim_mask.float(), size=recon_outputs["mask_logits"].shape[-2:], mode="nearest").squeeze(1).bool())
                small_labels = small_labels.masked_fill(~small_mask, cfg["ignore_index"])
                loss_rsc = criterion_rsc(recon_outputs["mask_logits"], small_labels)
            else:
                loss_r = seg_logits.sum() * 0.0
                loss_rsc = seg_logits.sum() * 0.0

            loss = (
                lambda_l * loss_l
                + lambda_e * loss_e
                + lambda_c * loss_c
                + lambda_r * loss_r
                + lambda_rsc * loss_rsc
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader_u) + i
            lr = update_lr(optimizer, cfg, iters, total_iters)
            update_ema(model, model_ema, iters, max_decay=ema_decay)

            meter.update(
                {
                    "loss": loss.item(),
                    "loss_l": loss_l.item(),
                    "loss_e": loss_e.item(),
                    "loss_c": loss_c.item(),
                    "loss_r": loss_r.item(),
                    "loss_rsc": loss_rsc.item(),
                }
            )

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_l", loss_l.item(), iters)
                writer.add_scalar("train/loss_e", loss_e.item(), iters)
                writer.add_scalar("train/loss_c", loss_c.item(), iters)
                writer.add_scalar("train/loss_r", loss_r.item(), iters)
                writer.add_scalar("train/loss_rsc", loss_rsc.item(), iters)
                writer.add_scalar("train/lr", lr, iters)

                if i < 5:
                    viz.push(
                        {
                            "img_l": (img_l_s[0], Visualizer.TENSOR),
                            "mask_l": (mask_l[0], Visualizer.SEGMENTATION),
                            "pred_l": (seg_logits.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                            "img_u": (mixed_u["img_s"][0], Visualizer.TENSOR),
                            "pseudo_u": (mixed_u["pseudo_label"][0], Visualizer.SEGMENTATION),
                            "pred_u": (seg_logits.argmax(dim=1)[img_l_s.shape[0]], Visualizer.SEGMENTATION),
                        }
                    )
                    viz.render(f"epoch_{epoch}_iter_{i}")
                    viz.reset()

            log_interval = max(1, len(trainloader_u) // 8)
            if rank == 0 and i % log_interval == 0:
                logger.info("Iters: %d, LR: %.7f, %s", i, lr, meter)

        val_cfg = dict(cfg)
        val_cfg.setdefault("eval_mode", "slide_window" if cfg["dataset"] == "cityscapes" else "original")
        val_cfg.setdefault("ignore_index", cfg.get("ignore_index", 255))
        mIoU, iou_class = validation_cpu(val_cfg, model, valloader)
        mIoU_ema, iou_class_ema = validation_cpu(val_cfg, model_ema, valloader)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [%d %s] IoU: %.2f, EMA: %.2f",
                    cls_idx,
                    CLASSES[cfg["dataset"]][cls_idx],
                    iou,
                    iou_class_ema[cls_idx],
                )
            logger.info("***** Evaluation ***** >>>> MeanIoU: %.2f, EMA: %.2f\n", mIoU, mIoU_ema)
            writer.add_scalar("eval/mIoU", mIoU, epoch)
            writer.add_scalar("eval/mIoU_ema", mIoU_ema, epoch)

        is_best = mIoU >= previous_best
        previous_best = max(previous_best, mIoU)
        previous_best_ema = max(previous_best_ema, mIoU_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch

        save_checkpoint(
            args,
            rank,
            model,
            optimizer,
            epoch,
            previous_best,
            best_epoch,
            model_ema=model_ema,
            previous_best_ema=previous_best_ema,
            best_epoch_ema=best_epoch_ema,
            extra={"segmind_bank": memory_bank.state_dict()},
            is_best=is_best,
        )


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg.setdefault("ignore_index", cfg.get("criterion", {}).get("kwargs", {}).get("ignore_index", 255))
    main(args, cfg)
