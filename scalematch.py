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

from dataset.semi_rs import SemiDataset as RemoteSemiDataset
from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.upernet import UperNet
from util.classes import CLASSES
from util.dist_helper import setup_distributed
from util.focal import FocalLoss
from util.ohem import ProbOhemCrossEntropy2d
from util.ssl_method_utils import (
    get_backbone_info,
    get_local_rank,
    get_model_kwargs,
    load_backbone_checkpoint,
    load_checkpoint_on_cpu,
    log_cuda_memory,
    save_checkpoint_to_disk,
)
from util.train_utils import (
    DictAverageMeter,
    confidence_weighted_loss,
    cutmix_img_,
    cutmix_mask,
)
from util.utils import count_params, init_log
from util.validation import validation_cpu as shared_validation_cpu


REMOTE_SENSING_DATASETS = {"pascal", "cityscapes", "iSAID", "vaihingen", "potsdam", "loveda"}
DEFAULT_IMG_SCALES = [0.5, 0.75, 1.0, 1.25]
DEFAULT_FEAT_S_SCALES = [0.75]
DEFAULT_FEAT_L_SCALES = [1.0, 1.25, 1.5]
OFFICIAL_WARM_UP = 10


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def get_debug_cfg(cfg):
    debug_cfg = cfg.get("debug", {})
    return {
        "enabled": debug_cfg.get("enabled", False),
        "class_stats_interval": max(int(debug_cfg.get("class_stats_interval", 50)), 1),
        "grad_stats_interval": max(int(debug_cfg.get("grad_stats_interval", 50)), 1),
        "viz_train_iters": max(int(debug_cfg.get("viz_train_iters", 5)), 0),
    }


def compute_masked_class_hist(labels, num_classes, valid_mask=None):
    labels = labels.detach().reshape(-1).long()
    if valid_mask is not None:
        valid_mask = valid_mask.detach().reshape(-1).bool()
        labels = labels[valid_mask]
    if labels.numel() == 0:
        return torch.zeros(num_classes, device=labels.device, dtype=torch.float32)
    valid_labels = labels[(labels >= 0) & (labels < num_classes)]
    if valid_labels.numel() == 0:
        return torch.zeros(num_classes, device=labels.device, dtype=torch.float32)
    return torch.bincount(valid_labels, minlength=num_classes).float()


def compute_class_ratio(labels, num_classes, valid_mask=None):
    hist = compute_masked_class_hist(labels, num_classes, valid_mask=valid_mask)
    total = hist.sum()
    if total <= 0:
        return hist
    return hist / total


def masked_agreement(pred_a, pred_b, valid_mask=None):
    same = pred_a.detach().eq(pred_b.detach())
    if valid_mask is not None:
        valid_mask = valid_mask.detach().bool()
        denom = valid_mask.sum().clamp(min=1).float()
        return (same & valid_mask).sum().float() / denom
    return same.float().mean()


def grad_norm(module):
    total = 0.0
    found = False
    for param in module.parameters():
        if param.grad is None:
            continue
        found = True
        total += float(param.grad.detach().norm().item() ** 2)
    if not found:
        return 0.0
    return total ** 0.5


def write_class_ratios(writer, prefix, ratios, class_names, step):
    for idx, class_name in enumerate(class_names):
        writer.add_scalar(f"{prefix}/{class_name}", ratios[idx].item(), step)


def compute_official_scalematch_total_loss(loss_x, loss_u_s1, loss_u_size, loss_u_w_fp):
    return (loss_x + 0.25 * loss_u_s1 + 0.25 * loss_u_size + 0.5 * loss_u_w_fp) / 2.0


def select_pseudo_logits_from_student_out(student_out, num_lb, epoch, warm_up):
    key = "pred_ori" if epoch < warm_up else "pred_joint"
    return student_out[key][num_lb:].detach()


def build_loader_guard_message(
    dataset_name,
    split_name,
    base_num_ids,
    effective_num_ids,
    loader_len,
    world_size,
    batch_size,
):
    return (
        f"ScaleMatch {dataset_name} {split_name} loader has zero batches under DDP: "
        f"base_num_ids={base_num_ids}, effective_num_ids={effective_num_ids}, "
        f"world_size={world_size}, batch_size={batch_size}, loader_len={loader_len}. "
        "Check that the split file is non-empty and try reducing --nproc_per_node "
        "or batch_size."
    )


def enable_ddp_static_graph(model, logger=None):
    if hasattr(model, "_set_static_graph"):
        model._set_static_graph()
        if logger is not None:
            logger.info("Enabled DDP static graph via _set_static_graph().")
    elif hasattr(model, "set_static_graph"):
        model.set_static_graph()
        if logger is not None:
            logger.info("Enabled DDP static graph via set_static_graph().")


def collect_debug_metrics(
    pred_u_w,
    student_out,
    pred_u_s,
    pred_x_joint,
    pred_x_ori,
    mask_u_w_cutmixed1,
    conf_u_w,
    conf_u_w_cutmixed1,
    valid_mask,
    ignore_mask_cutmixed1,
    ignore_index,
    conf_thresh,
    num_lb,
    nclass,
):
    teacher_pred = pred_u_w.detach().argmax(dim=1)
    student_ori_u = student_out["pred_ori"][num_lb:].detach().argmax(dim=1)
    student_joint_u = student_out["pred_joint"][num_lb:].detach().argmax(dim=1)
    strong_pred = pred_u_s.detach().argmax(dim=1)

    accepted_valid_mask = valid_mask & (conf_u_w >= conf_thresh)
    strong_valid_mask = (ignore_mask_cutmixed1 != ignore_index) & (conf_u_w_cutmixed1 >= conf_thresh)

    return {
        "teacher_vs_student_ori_agreement": masked_agreement(teacher_pred, student_ori_u, valid_mask),
        "teacher_vs_student_joint_agreement": masked_agreement(teacher_pred, student_joint_u, valid_mask),
        "student_joint_vs_ori_agreement": masked_agreement(student_joint_u, student_ori_u, valid_mask),
        "strong_vs_pseudo_agreement": masked_agreement(strong_pred, mask_u_w_cutmixed1, strong_valid_mask),
        "conf_teacher_pseudo": conf_u_w.mean(),
        "conf_student_ori_u": student_out["pred_ori"][num_lb:].detach().softmax(dim=1).amax(dim=1).mean(),
        "conf_student_joint_u": student_out["pred_joint"][num_lb:].detach().softmax(dim=1).amax(dim=1).mean(),
        "conf_student_strong": pred_u_s.detach().softmax(dim=1).amax(dim=1).mean(),
        "pseudo_ratio": compute_class_ratio(teacher_pred, nclass, valid_mask),
        "accepted_pseudo_ratio": compute_class_ratio(teacher_pred, nclass, accepted_valid_mask),
        "student_joint_ratio": compute_class_ratio(student_joint_u, nclass, valid_mask),
        "student_ori_ratio": compute_class_ratio(student_ori_u, nclass, valid_mask),
        "strong_ratio": compute_class_ratio(strong_pred, nclass, strong_valid_mask),
        "labeled_joint_ratio": compute_class_ratio(pred_x_joint.detach().argmax(dim=1), nclass),
        "labeled_ori_ratio": compute_class_ratio(pred_x_ori.detach().argmax(dim=1), nclass),
    }


def flip_batch(x):
    return x.flip(0)


def build_same_batch_cutmix_targets(
    img_u_s,
    cutmix_box,
    pseudo_mask,
    pseudo_conf,
    ignore_mask,
    pseudo_mask_mix=None,
    pseudo_conf_mix=None,
    ignore_mask_mix=None,
):
    img_u_s_mix = flip_batch(img_u_s)
    pseudo_mask_mix = flip_batch(pseudo_mask if pseudo_mask_mix is None else pseudo_mask_mix)
    pseudo_conf_mix = flip_batch(pseudo_conf if pseudo_conf_mix is None else pseudo_conf_mix)
    ignore_mask_mix = flip_batch(ignore_mask if ignore_mask_mix is None else ignore_mask_mix)

    cutmix_img_(img_u_s, img_u_s_mix, cutmix_box)
    pseudo_mask_cutmixed = cutmix_mask(pseudo_mask, pseudo_mask_mix, cutmix_box)
    pseudo_conf_cutmixed = cutmix_mask(pseudo_conf, pseudo_conf_mix, cutmix_box)
    ignore_mask_cutmixed = cutmix_mask(ignore_mask, ignore_mask_mix, cutmix_box)
    return pseudo_mask_cutmixed, pseudo_conf_cutmixed, ignore_mask_cutmixed


def get_scalematch_dataset_cls(dataset_name):
    if dataset_name not in REMOTE_SENSING_DATASETS:
        raise ValueError(f"Unsupported dataset for scalematch: {dataset_name}.")
    return RemoteSemiDataset, "semi_rs"


def get_scalematch_recipe(cfg):
    return {
        "img_scales": cfg.get("img_scales", DEFAULT_IMG_SCALES),
        "feat_s_scales": cfg.get("feat_s_scales", DEFAULT_FEAT_S_SCALES),
        "feat_l_scales": cfg.get("feat_l_scales", DEFAULT_FEAT_L_SCALES),
        "conf_thresh": cfg.get("conf_thresh", 0.95),
        "warm_up": cfg.get("warm_up", OFFICIAL_WARM_UP),
    }


def build_scalematch_model(cfg):
    model_kwargs = get_model_kwargs(cfg)
    _, backbone_version = get_backbone_info(cfg)
    model_name = cfg.get("model", "dpt").lower()

    if model_name == "dpt":
        model = DPT(
            **model_kwargs,
            backbone_version=backbone_version,
            enable_scalematch=True,
        )
    elif model_name == "upernet":
        model = UperNet(
            **model_kwargs,
            backbone_version=backbone_version,
            enable_scalematch=True,
        )
    else:
        raise ValueError(
            f"Unsupported ScaleMatch model '{cfg.get('model')}'. This port currently supports only 'dpt' and 'upernet'."
        )
    return model, backbone_version


def get_parser():
    parser = argparse.ArgumentParser(
        description="ScaleMatch rebuilt on top of the supervised.py scaffold"
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
    amp = cfg.get("amp", False)

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model, backbone_version = build_scalematch_model(cfg)
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
        find_unused_parameters=False,
    )
    enable_ddp_static_graph(model, logger=logger)
    log_cuda_memory(logger, rank, "after_ddp_wrap", local_rank=local_rank, save_path=args.save_path)

    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "FocalLoss":
        criterion_l = FocalLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    else:
        raise NotImplementedError("%s criterion is not implemented" % cfg["criterion"]["name"])

    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)

    SemiDataset, dataset_loader_name = get_scalematch_dataset_cls(cfg["dataset"])
    if rank == 0:
        logger.info("ScaleMatch dataset loader: %s for %s", dataset_loader_name, cfg["dataset"])

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
    valset = ValDataset(cfg["dataset"], cfg["data_root"], "val", ignore_value=cfg["ignore_index"])

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

    recipe = get_scalematch_recipe(cfg)
    img_scales = recipe["img_scales"]
    feat_s_scales = recipe["feat_s_scales"]
    feat_l_scales = recipe["feat_l_scales"]
    conf_thresh = recipe["conf_thresh"]
    warm_up = recipe["warm_up"]

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best = 0.0
    best_epoch = 0
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
        best_epoch = checkpoint.get("best_epoch", 0)
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
        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        model.train()
        log_interval = max(len(trainloader_u) // 8, 1)

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
            (img_u_w_mix, img_u_s1_mix, _, ignore_mask_mix, _, _),
        ) in enumerate(loader):
            random_scale = random.choice(img_scales)
            feature_scale = random.choice(
                feat_s_scales if random_scale > 1 else feat_l_scales
            )

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, ignore_mask = img_u_s1.cuda(), ignore_mask.cuda()
            cutmix_box1 = cutmix_box1.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix = img_u_s1_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix, scale_factor=None)
                if isinstance(pred_u_w_mix, dict):
                    pred_u_w_mix = pred_u_w_mix["pred_ori"]
                pred_u_w_mix = pred_u_w_mix.detach()
                conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)
            model.train()

            cutmix_img_(img_u_s1, img_u_s1_mix, cutmix_box1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp):
                num_lb = img_x.shape[0]
                student_out = model(
                    torch.cat((img_x, img_u_w)),
                    scale_factor=random_scale,
                    feature_scale=feature_scale,
                )
                pred_u_w = select_pseudo_logits_from_student_out(
                    student_out=student_out,
                    num_lb=num_lb,
                    epoch=epoch,
                    warm_up=warm_up,
                )
                conf_u_w, mask_u_w = pred_u_w.softmax(dim=1).max(dim=1)

                mask_u_w_cutmixed1 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box1)
                conf_u_w_cutmixed1 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box1)
                ignore_mask_cutmixed1 = cutmix_mask(ignore_mask, ignore_mask_mix, cutmix_box1)

                pred_u_s = model(img_u_s1, scale_factor=None)
                if isinstance(pred_u_s, dict):
                    pred_u_s = pred_u_s["pred_ori"]

                pred_x_joint = student_out["pred_joint"][:num_lb]
                pred_u_w_scale = student_out["pred_size"][num_lb:]
                pred_u_w_fp = student_out["pred_fp"][num_lb:]

                loss_x = criterion_l(pred_x_joint, mask_x)

                loss_u_s1 = criterion_u(pred_u_s, mask_u_w_cutmixed1)
                loss_u_s1 = confidence_weighted_loss(
                    loss_u_s1,
                    conf_u_w_cutmixed1,
                    ignore_mask_cutmixed1,
                    cfg["ignore_index"],
                    conf_thresh=conf_thresh,
                )

                loss_u_size = criterion_u(pred_u_w_scale, mask_u_w)
                loss_u_size = confidence_weighted_loss(
                    loss_u_size,
                    conf_u_w,
                    ignore_mask,
                    cfg["ignore_index"],
                    conf_thresh=conf_thresh,
                )

                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = confidence_weighted_loss(
                    loss_u_w_fp,
                    conf_u_w,
                    ignore_mask,
                    cfg["ignore_index"],
                    conf_thresh=conf_thresh,
                )

                total_loss = compute_official_scalematch_total_loss(
                    loss_x, loss_u_s1, loss_u_size, loss_u_w_fp
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            valid_mask = ignore_mask != cfg["ignore_index"]
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

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                for key, value in log_avg.avgs.items():
                    writer.add_scalar(f"train/{key}", value, iters)

            if (i % log_interval == 0) and (rank == 0):
                logger.info(f"Iters: {i}, {log_avg}")
                log_avg.reset()

        val_cfg = dict(cfg)
        val_cfg.setdefault("eval_mode", "slide_window" if cfg["dataset"] == "cityscapes" else "original")
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
                writer.add_scalar(f"eval/{CLASSES[cfg['dataset']][i]}_IoU", iou, epoch)

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
    main(args, cfg)
