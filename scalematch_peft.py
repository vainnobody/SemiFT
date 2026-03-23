"""ScaleMatch + configurable PEFT fine-tuning."""

import argparse
from datetime import datetime
import logging
import os
import pprint
import random
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.val import ValDataset
from fixmatch_peft import apply_peft, resolve_peft_cfg, show_trainable_parameters
from scalematch import (
    REMOTE_SENSING_DATASETS,
    build_scalematch_model,
    build_same_batch_cutmix_targets,
    collect_debug_metrics,
    compute_official_scalematch_total_loss,
    enable_ddp_static_graph,
    flip_batch,
    get_debug_cfg,
    get_scalematch_dataset_cls,
    get_scalematch_recipe,
    grad_norm,
    select_pseudo_logits_from_student_out,
    write_class_ratios,
)
from util.classes import CLASSES
from util.dist_helper import setup_distributed
from util.ssl_method_utils import get_local_rank, load_checkpoint_on_cpu, save_checkpoint_to_disk, log_cuda_memory, checkpoint_to_cpu
from util.focal import FocalLoss
from util.ohem import ProbOhemCrossEntropy2d
from util.train_utils import DictAverageMeter, confidence_weighted_loss
from util.utils import count_params, init_log

try:
    from supervised import evaluate
except ImportError:  # pragma: no cover - test stubs may only expose validation_cpu
    def evaluate(*args, **kwargs):
        raise ImportError("supervised.evaluate is unavailable in the current environment")


def get_parser():
    parser = argparse.ArgumentParser(
        description="ScaleMatch + configurable PEFT for Semi-Supervised Semantic Segmentation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--peft-method", type=str, default=None)
    parser.add_argument(
        "--peft-target-modules",
        nargs="+",
        default=None,
        help="Override PEFT target modules. Pass one or more suffixes, or a single regex string.",
    )
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        help="Freeze backbone parameters before applying PEFT.",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Keep backbone parameters trainable outside PEFT adapters.",
    )
    parser.set_defaults(freeze_backbone=None)
    return parser.parse_args()


def build_model(cfg, peft_cfg, logger=None, rank=0):
    model, backbone_version = build_scalematch_model(cfg)

    backbone_ckpt_path = f'./pretrained/{cfg["backbone"]}.pth'
    if logger is not None and rank == 0:
        logger.info(f"Backbone version: {backbone_version}")
        logger.info(f"Backbone checkpoint: {backbone_ckpt_path}")

    state_dict = torch.load(backbone_ckpt_path, map_location="cpu")
    load_result = model.backbone.load_state_dict(state_dict)
    if logger is not None and rank == 0:
        logger.info(
            "Backbone load result | missing_keys=%d unexpected_keys=%d",
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )
        if load_result.missing_keys:
            logger.info(f"Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.info(f"Unexpected keys: {load_result.unexpected_keys}")
        logger.info(f"Loaded {backbone_version} backbone weights successfully")

    if peft_cfg.get("freeze_backbone", True) or cfg.get("lock_backbone", False):
        model.lock_backbone()

    model = apply_peft(model, peft_cfg, cfg)
    return model, backbone_version


def build_optimizer(model, cfg):
    trainable_backbone_params = []
    trainable_non_backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            trainable_backbone_params.append(param)
        else:
            trainable_non_backbone_params.append(param)

    return AdamW(
        [
            {"params": trainable_backbone_params, "lr": cfg["lr"]},
            {
                "params": trainable_non_backbone_params,
                "lr": cfg["lr"] * cfg["lr_multi"],
            },
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )


def get_reference_eval_settings(cfg, model_noddp):
    eval_mode = "sliding_window" if cfg["dataset"] == "cityscapes" else "original"
    multiplier = (
        None
        if cfg.get("model", "dpt").lower() == "upernet"
        else model_noddp.backbone.patch_size
    )
    return eval_mode, multiplier


def get_reference_eval_mode_from_cfg(cfg):
    return "sliding_window" if cfg["dataset"] == "cityscapes" else "original"


def main(args, cfg):
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    peft_cfg = resolve_peft_cfg(cfg, args)
    amp = cfg.get("amp", False)
    debug_cfg = get_debug_cfg(cfg)
    debug_enabled = debug_cfg["enabled"]
    viz = None

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        all_args.setdefault("eval_mode", get_reference_eval_mode_from_cfg(cfg))
        logger.info("{}\n".format(pprint.pformat(all_args)))
        logger.info(
            "Running ScaleMatch + PEFT with method=%s, target_modules=%s, freeze_backbone=%s",
            peft_cfg["method"],
            peft_cfg["target_modules"],
            peft_cfg["freeze_backbone"],
        )
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)
        if debug_enabled:
            from util.viz import Visualizer

            filename = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz = Visualizer(
                save_dir=f"./viz/{filename}_scalematch_peft", dataset=cfg["dataset"]
            )

    cudnn.enabled = True
    cudnn.benchmark = True

    model, _ = build_model(cfg, peft_cfg, logger=logger, rank=rank)
    optimizer = build_optimizer(model, cfg)

    if rank == 0:
        logger.info("Total params: {:.1f}M".format(count_params(model)))
        logger.info("Encoder params: {:.1f}M".format(count_params(model.backbone)))
        logger.info("Decoder params: {:.1f}M\n".format(count_params(model.head)))
        logger.info(
            "Optimizer: AdamW | lr_backbone=%.8f lr_head=%.8f",
            cfg["lr"],
            cfg["lr"] * cfg["lr_multi"],
        )
        show_trainable_parameters(model, logger)

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
    enable_ddp_static_graph(model, logger=logger if rank == 0 else None)
    model_noddp = model.module

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

    ignore_index = cfg.get("ignore_index", 255)
    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)

    SemiDataset, dataset_loader_name = get_scalematch_dataset_cls(cfg["dataset"])
    if rank == 0:
        logger.info(
            f"ScaleMatch dataset loader: {dataset_loader_name} for {cfg['dataset']}"
        )
        if "epoch_repeat_factor" in cfg:
            logger.warning(
                "ScaleMatch PEFT ignores deprecated config key epoch_repeat_factor=%s; "
                "the loader length now follows the base dataset semantics.",
                cfg["epoch_repeat_factor"],
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
        cfg["dataset"],
        cfg["data_root"],
        "val",
        ignore_value=ignore_index,
    )

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg.get("workers", 4),
        drop_last=True,
        sampler=trainsampler_l,
    )

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg.get("workers", 4),
        drop_last=True,
        sampler=trainsampler_u,
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=cfg.get("val_workers", 1),
        drop_last=False,
        sampler=valsampler,
    )

    recipe = get_scalematch_recipe(cfg)
    img_scales = recipe["img_scales"]
    feat_s_scales = recipe["feat_s_scales"]
    feat_l_scales = recipe["feat_l_scales"]
    conf_thresh = recipe["conf_thresh"]
    warm_up = recipe["warm_up"]

    total_epochs = cfg["epochs"]
    total_iters = len(trainloader_u) * total_epochs
    previous_best = 0.0
    best_epoch = 0
    epoch = -1
    eta_seconds = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    latest_path = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(latest_path):
        log_cuda_memory(logger, rank, "before_resume_load", save_path=args.save_path)
        checkpoint = load_checkpoint_on_cpu(latest_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        log_cuda_memory(logger, rank, "after_resume_load", save_path=args.save_path)
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        best_epoch = checkpoint.get("best_epoch", 0)

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    for epoch in range(epoch + 1, total_epochs):
        start_time = time.time()

        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f} @epoch-{:}, ETA: {:.2f}M".format(
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    previous_best,
                    best_epoch,
                    eta_seconds / 60,
                )
            )

        log_avg = DictAverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        model.train()

        log_interval = max(len(trainloader_u) // 8, 1)

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
        ) in enumerate(loader):
            random_scale = random.choice(img_scales)
            feature_scale = random.choice(
                feat_s_scales if random_scale > 1 else feat_l_scales
            )

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, ignore_mask = img_u_s1.cuda(), ignore_mask.cuda()
            cutmix_box1 = cutmix_box1.cuda()

            iters = epoch * len(trainloader_u) + i

            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp):
                    pred_u_w_mix = model_noddp(img_u_w, scale_factor=None)
                    if isinstance(pred_u_w_mix, dict):
                        pred_u_w_mix = pred_u_w_mix["pred_ori"]
                    conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)

                    teacher_out = model_noddp(
                        img_u_w,
                        scale_factor=random_scale,
                        feature_scale=feature_scale,
                    )
                    pred_u_w = select_pseudo_logits_from_student_out(
                        {
                            "pred_ori": teacher_out["pred_ori"],
                            "pred_joint": teacher_out["pred_joint"],
                        },
                        num_lb=0,
                        epoch=epoch,
                        warm_up=warm_up,
                    )
                    conf_u_w, mask_u_w = pred_u_w.detach().softmax(dim=1).max(dim=1)
            model.train()
            optimizer.zero_grad()

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = (
                build_same_batch_cutmix_targets(
                    img_u_s=img_u_s1,
                    cutmix_box=cutmix_box1,
                    pseudo_mask=mask_u_w,
                    pseudo_conf=conf_u_w,
                    ignore_mask=ignore_mask,
                    pseudo_mask_mix=mask_u_w_mix,
                    pseudo_conf_mix=conf_u_w_mix,
                )
            )

            with torch.cuda.amp.autocast(enabled=amp):
                num_lb = img_x.shape[0]
                student_out = model(
                    torch.cat((img_x, img_u_w)),
                    scale_factor=random_scale,
                    feature_scale=feature_scale,
                )
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

                pred_x_joint = student_out["pred_joint"][:num_lb]
                pred_x_ori = student_out["pred_ori"][:num_lb]
                pred_u_w_scale = student_out["pred_size"][num_lb:]
                pred_u_w_fp = student_out["pred_fp"][num_lb:]

                loss_x_joint = criterion_l(pred_x_joint, mask_x)
                loss_x_ori = criterion_l(pred_x_ori, mask_x)
                loss_x = loss_x_joint
                _ = loss_x_ori

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

                total_loss = compute_official_scalematch_total_loss(
                    loss_x, loss_u_s1, loss_u_size, loss_u_w_fp
                )

            scaler.scale(total_loss).backward()

            if debug_enabled and rank == 0 and (i % debug_cfg["grad_stats_interval"] == 0):
                writer.add_scalar(
                    "debug/grad/scale_attn", grad_norm(model_noddp.scale_attn), iters
                )
                writer.add_scalar(
                    "debug/grad/se_block", grad_norm(model_noddp.se_block), iters
                )
                writer.add_scalar(
                    "debug/grad/rwkv_layers", grad_norm(model_noddp.rwkv_layers), iters
                )
                writer.add_scalar("debug/grad/head", grad_norm(model_noddp.head), iters)

            scaler.step(optimizer)
            scaler.update()

            valid_mask = ignore_mask != ignore_index
            mask_ratio = ((conf_u_w >= conf_thresh) & valid_mask).sum().float() / valid_mask.sum().clamp(min=1.0)

            log_avg.update(
                {
                    "Total_loss": total_loss,
                    "Loss_x": loss_x,
                    "Loss_u_s": loss_u_s1,
                    "Loss_u_scale": loss_u_size,
                    "Loss_w_fp_scale": loss_u_w_fp,
                    "Mask_ratio": mask_ratio,
                }
            )

            if debug_enabled and rank == 0 and (i % debug_cfg["class_stats_interval"] == 0):
                debug_metrics = collect_debug_metrics(
                    pred_u_w=pred_u_w,
                    student_out=student_out,
                    pred_u_s=pred_u_s,
                    pred_x_joint=pred_x_joint,
                    pred_x_ori=pred_x_ori,
                    mask_u_w_cutmixed1=mask_u_w_cutmixed1,
                    conf_u_w=conf_u_w,
                    conf_u_w_cutmixed1=conf_u_w_cutmixed1,
                    valid_mask=valid_mask,
                    ignore_mask_cutmixed1=ignore_mask_cutmixed1,
                    ignore_index=ignore_index,
                    conf_thresh=conf_thresh,
                    num_lb=num_lb,
                    nclass=cfg["nclass"],
                )
                for metric_name in (
                    "teacher_vs_student_ori_agreement",
                    "teacher_vs_student_joint_agreement",
                    "student_joint_vs_ori_agreement",
                    "strong_vs_pseudo_agreement",
                    "conf_teacher_pseudo",
                    "conf_student_ori_u",
                    "conf_student_joint_u",
                    "conf_student_strong",
                ):
                    writer.add_scalar(f"debug/{metric_name}", debug_metrics[metric_name], iters)
                write_class_ratios(
                    writer,
                    "debug/pseudo_ratio",
                    debug_metrics["pseudo_ratio"],
                    CLASSES[cfg["dataset"]],
                    iters,
                )
                write_class_ratios(
                    writer,
                    "debug/accepted_pseudo_ratio",
                    debug_metrics["accepted_pseudo_ratio"],
                    CLASSES[cfg["dataset"]],
                    iters,
                )
                write_class_ratios(
                    writer,
                    "debug/student_joint_ratio",
                    debug_metrics["student_joint_ratio"],
                    CLASSES[cfg["dataset"]],
                    iters,
                )
                write_class_ratios(
                    writer,
                    "debug/student_ori_ratio",
                    debug_metrics["student_ori_ratio"],
                    CLASSES[cfg["dataset"]],
                    iters,
                )
                write_class_ratios(
                    writer,
                    "debug/strong_ratio",
                    debug_metrics["strong_ratio"],
                    CLASSES[cfg["dataset"]],
                    iters,
                )
                write_class_ratios(
                    writer,
                    "debug/labeled_joint_ratio",
                    debug_metrics["labeled_joint_ratio"],
                    CLASSES[cfg["dataset"]],
                    iters,
                )
                write_class_ratios(
                    writer,
                    "debug/labeled_ori_ratio",
                    debug_metrics["labeled_ori_ratio"],
                    CLASSES[cfg["dataset"]],
                    iters,
                )

            if debug_enabled and rank == 0 and viz is not None and i < debug_cfg["viz_train_iters"]:
                viz.push(
                    {
                        "img_x": (img_x[0], viz.TENSOR),
                        "mask_x": (mask_x[0], viz.SEGMENTATION),
                        "pred_x_ori": (pred_x_ori.argmax(dim=1)[0], viz.SEGMENTATION),
                        "pred_x_joint": (pred_x_joint.argmax(dim=1)[0], viz.SEGMENTATION),
                        "img_u_w": (img_u_w[0], viz.TENSOR),
                        "pseudo_u_w": (mask_u_w[0], viz.SEGMENTATION),
                        "pred_u_w_ori_student": (
                            student_out["pred_ori"][num_lb:].argmax(dim=1)[0],
                            viz.SEGMENTATION,
                        ),
                        "pred_u_w_joint_student": (
                            student_out["pred_joint"][num_lb:].argmax(dim=1)[0],
                            viz.SEGMENTATION,
                        ),
                        "img_u_s1": (img_u_s1[0], viz.TENSOR),
                        "mask_cutmix": (mask_u_w_cutmixed1[0], viz.SEGMENTATION),
                        "pred_u_s": (pred_u_s.argmax(dim=1)[0], viz.SEGMENTATION),
                        "pred_u_w_scale": (pred_u_w_scale.argmax(dim=1)[0], viz.SEGMENTATION),
                        "pred_u_w_fp": (pred_u_w_fp.argmax(dim=1)[0], viz.SEGMENTATION),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if (i % log_interval == 0) and (rank == 0):
                for key, value in log_avg.avgs.items():
                    writer.add_scalar(
                        "train/" + key,
                        value.item() if torch.is_tensor(value) else value,
                        iters,
                    )
                logger.info(f"Iters: {i}, " + str(log_avg))
                log_avg.reset()

        eval_mode, multiplier = get_reference_eval_settings(cfg, model_noddp)
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=multiplier)

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
            for idx, iou in enumerate(iou_class):
                writer.add_scalar("eval/%s_IoU" % CLASSES[cfg["dataset"]][idx], iou, epoch)

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
            save_checkpoint_to_disk(
                checkpoint,
                latest_path,
                os.path.join(args.save_path, "best.pth"),
                is_best=is_best,
            )

        eta_seconds = (total_epochs - (epoch + 1)) * (time.time() - start_time)


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
