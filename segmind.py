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
    segmind_contrastive_loss,
)


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def build_entropy_targets(teacher_entropy_l, mixed_entropy):
    return torch.cat((teacher_entropy_l, mixed_entropy), dim=0)


def needs_pseudo_branch(segmind_cfg):
    return any(
        float(segmind_cfg.get(key, 1.0)) != 0.0
        for key in ("lambda_l", "lambda_e", "lambda_r", "lambda_rsc", "lambda_c")
    )


def validate_loss_weights(segmind_cfg):
    if not needs_pseudo_branch(segmind_cfg):
        raise ValueError("At least one SegMind loss weight must be non-zero.")

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
    segmind_cfg = cfg.setdefault("segmind", {})
    validate_loss_weights(segmind_cfg)
    use_pseudo_branch = needs_pseudo_branch(segmind_cfg)

    logger, rank, _, writer = build_logger_and_runtime(args, cfg)
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
    criterion_rsc = nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=cfg["ignore_index"],
    ).cuda(local_rank)

    trainloader_l, trainloader_u, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader_u) * cfg["epochs"]

    state = maybe_load_checkpoint(
        args,
        model,
        optimizer,
        model_ema=model_ema,
        logger=logger,
        rank=rank,
    )
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
    mask_gap = segmind_cfg.get("mask_gap", 4)
    mask_rate = segmind_cfg.get("mask_rate", 0.25)
    epoch_pre = segmind_cfg.get("epoch_pre", max(cfg["epochs"] // 2, 1))
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
        model_ema.train()
        meter = DictAverageMeter()
        for i, ((img_l_w, img_l_s, mask_l), (img_u_w, img_u_s, ignore_u)) in enumerate(zip(trainloader_l, trainloader_u)):
            img_l_w = img_l_w.cuda(local_rank)
            img_l_s = img_l_s.cuda(local_rank)
            mask_l = mask_l.cuda(local_rank)
            img_u_w = img_u_w.cuda(local_rank)
            img_u_s = img_u_s.cuda(local_rank)
            ignore_u = ignore_u.cuda(local_rank)
            del ignore_u

            zero = img_l_w.sum() * 0.0
            loss_x = zero
            loss_e = zero
            loss_c = zero
            loss_r = zero
            loss_rsc = zero
            logits_l = None

            if use_pseudo_branch:
                teacher_logits, pseudo_logit, pseudo_label, teacher_entropy_u = gather_pseudo_from_teacher(model_ema, img_l_w, img_u_w)
                teacher_prob = teacher_logits.softmax(dim=1)
                teacher_entropy_l = torch.sum(
                    -teacher_prob[: img_l_w.shape[0]] * torch.log(teacher_prob[: img_l_w.shape[0]].clamp_min(1e-8)),
                    dim=1,
                )
                mixed_u = class_mix_batch(
                    img_u_w=img_u_w,
                    img_s=img_u_s,
                    pseudo_label=pseudo_label,
                    pseudo_logit=pseudo_logit,
                    entropy=teacher_entropy_u,
                )

                strong_inputs = torch.cat((img_l_s, mixed_u["img_s"]), dim=0)
                weak_inputs = torch.cat((img_l_w, mixed_u["img_w"]), dim=0)
                label_all = torch.cat((mask_l, mixed_u["pseudo_label"]), dim=0)

                strong_outputs = model(strong_inputs, return_aux=True)
                seg_logits = strong_outputs["seg_logits"]
                prob_all = seg_logits.softmax(dim=1)
                logits_l, _ = seg_logits.split([img_l_s.shape[0], mixed_u["img_s"].shape[0]], dim=0)
                loss_x = criterion_l(seg_logits, label_all)

                if lambda_e != 0.0:
                    student_entropy = torch.sum(-prob_all * torch.log(prob_all.clamp_min(1e-8)), dim=1)
                    teacher_entropy_all = build_entropy_targets(teacher_entropy_l, mixed_u["entropy"])
                    loss_e = F.mse_loss(student_entropy, teacher_entropy_all)

                if lambda_c != 0.0:
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

                if (lambda_r != 0.0 or lambda_rsc != 0.0) and epoch + 1 <= epoch_pre:
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
                    recon_img = F.interpolate(
                        recon_outputs["recon_img"],
                        size=weak_inputs.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    hidden_region = ~mim_mask.bool().expand_as(recon_img)
                    if hidden_region.any():
                        loss_r = criterion_r(recon_img[hidden_region], weak_inputs[hidden_region])
                    else:
                        loss_r = recon_img.sum() * 0.0
                    hidden_labels = label_all.masked_fill(mim_mask.squeeze(1).bool(), cfg["ignore_index"])
                    loss_rsc_map = criterion_rsc(recon_outputs["seg_logits"], hidden_labels)
                    valid_hidden = hidden_labels != cfg["ignore_index"]
                    if valid_hidden.any():
                        loss_rsc = loss_rsc_map[valid_hidden].mean()
                    else:
                        loss_rsc = loss_rsc_map.sum() * 0.0

            loss = lambda_l * loss_x + lambda_e * loss_e + lambda_c * loss_c + lambda_r * loss_r + lambda_rsc * loss_rsc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loader_len = len(trainloader_u)
            iters = epoch * current_loader_len + i
            lr = update_lr(optimizer, cfg, iters, total_iters)
            if use_pseudo_branch:
                update_ema(model, model_ema, iters, max_decay=ema_decay)

            meter.update(
                {
                    "loss": loss.item(),
                    "loss_x": loss_x.item(),
                    "loss_e": loss_e.item(),
                    "loss_c": loss_c.item(),
                    "loss_r": loss_r.item(),
                    "loss_rsc": loss_rsc.item(),
                }
            )

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                if use_pseudo_branch:
                    writer.add_scalar("train/pseudo_conf", mixed_u["pseudo_logit"].mean().item(), iters)
                writer.add_scalar("train/loss_e", loss_e.item(), iters)
                writer.add_scalar("train/loss_c", loss_c.item(), iters)
                writer.add_scalar("train/loss_r", loss_r.item(), iters)
                writer.add_scalar("train/loss_rsc", loss_rsc.item(), iters)
                writer.add_scalar("train/lr", lr, iters)

                if i < 5:
                    viz.push(
                        {
                            "img_l_w": (img_l_w[0], Visualizer.TENSOR),
                            "mask_l": (mask_l[0], Visualizer.SEGMENTATION),
                            "pred_l_w": (logits_l.argmax(dim=1)[0], Visualizer.SEGMENTATION) if logits_l is not None else (mask_l[0], Visualizer.SEGMENTATION),
                        }
                    )
                    if use_pseudo_branch:
                        viz.push(
                            {
                                "img_l": (img_l_s[0], Visualizer.TENSOR),
                                "pred_joint_l": (seg_logits.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                                "img_u": (mixed_u["img_s"][0], Visualizer.TENSOR),
                                "pseudo_u": (mixed_u["pseudo_label"][0], Visualizer.SEGMENTATION),
                                "pred_u": (seg_logits.argmax(dim=1)[img_l_s.shape[0]], Visualizer.SEGMENTATION),
                            }
                        )
                    viz.render(f"epoch_{epoch}_iter_{i}")
                    viz.reset()

            log_interval = max(1, current_loader_len // 8)
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
