"""
ScaleMatch: Multi-scale semi-supervised semantic segmentation.

Ported from the official ScaleMatch training logic onto the SemiFT DPT backbone.
"""

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

from dataset.semi import SemiDataset as NaturalSemiDataset
from dataset.semi_rs import SemiDataset as RemoteSemiDataset
from dataset.val import ValDataset
from model.semseg.dpt_scalematch import DPT_ScaleMatch
from supervised import validation_cpu
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.focal import FocalLoss
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
from util.train_utils import (
    DictAverageMeter,
    confidence_weighted_loss,
    cutmix_img_,
    cutmix_mask,
)


NATURAL_IMAGE_DATASETS = {"pascal", "cityscapes"}
REMOTE_SENSING_DATASETS = {"iSAID", "vaihingen", "potsdam", "loveda"}
DEFAULT_IMG_SCALES = [0.5, 0.75, 1.0, 1.25]
DEFAULT_FEAT_S_SCALES = [0.75]
DEFAULT_FEAT_L_SCALES = [1.0, 1.25, 1.5]
OFFICIAL_WARM_UP = 10
MODEL_CONFIGS = {
    "small": {
        "encoder_size": "small",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "base": {
        "encoder_size": "base",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "large": {
        "encoder_size": "large",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "giant": {
        "encoder_size": "giant",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


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
    hist = torch.bincount(valid_labels, minlength=num_classes).float()
    return hist


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
    return total**0.5


def write_class_ratios(writer, prefix, ratios, class_names, step):
    for idx, class_name in enumerate(class_names):
        writer.add_scalar(f"{prefix}/{class_name}", ratios[idx].item(), step)


def compute_official_scalematch_total_loss(
    loss_x, loss_u_s1, loss_u_size, loss_u_w_fp
):
    return (loss_x + 0.25 * loss_u_s1 + 0.25 * loss_u_size + 0.5 * loss_u_w_fp) / 2.0


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
    strong_valid_mask = (ignore_mask_cutmixed1 != ignore_index) & (
        conf_u_w_cutmixed1 >= conf_thresh
    )

    metrics = {
        "teacher_vs_student_ori_agreement": masked_agreement(
            teacher_pred, student_ori_u, valid_mask
        ),
        "teacher_vs_student_joint_agreement": masked_agreement(
            teacher_pred, student_joint_u, valid_mask
        ),
        "student_joint_vs_ori_agreement": masked_agreement(
            student_joint_u, student_ori_u, valid_mask
        ),
        "strong_vs_pseudo_agreement": masked_agreement(
            strong_pred, mask_u_w_cutmixed1, strong_valid_mask
        ),
        "conf_teacher_pseudo": conf_u_w.mean(),
        "conf_student_ori_u": student_out["pred_ori"][num_lb:]
        .detach()
        .softmax(dim=1)
        .amax(dim=1)
        .mean(),
        "conf_student_joint_u": student_out["pred_joint"][num_lb:]
        .detach()
        .softmax(dim=1)
        .amax(dim=1)
        .mean(),
        "conf_student_strong": pred_u_s.detach().softmax(dim=1).amax(dim=1).mean(),
        "pseudo_ratio": compute_class_ratio(teacher_pred, nclass, valid_mask),
        "accepted_pseudo_ratio": compute_class_ratio(
            teacher_pred, nclass, accepted_valid_mask
        ),
        "student_joint_ratio": compute_class_ratio(student_joint_u, nclass, valid_mask),
        "student_ori_ratio": compute_class_ratio(student_ori_u, nclass, valid_mask),
        "strong_ratio": compute_class_ratio(strong_pred, nclass, strong_valid_mask),
        "labeled_joint_ratio": compute_class_ratio(
            pred_x_joint.detach().argmax(dim=1), nclass
        ),
        "labeled_ori_ratio": compute_class_ratio(
            pred_x_ori.detach().argmax(dim=1), nclass
        ),
    }
    return metrics


class ScaleMatchRemoteSemiDataset(RemoteSemiDataset):
    """Remote-sensing dataset wrapper with configurable epoch repeat factor.

    The base remote dataset samples randomly in ``__getitem__`` and multiplies
    ``__len__`` by 50, which makes ScaleMatch epochs prohibitively long because
    this trainer already performs several heavy forward passes per iteration.
    We keep the random sampling behavior but use a much smaller configurable
    repeat factor for the effective epoch length.
    """

    def __init__(self, *args, epoch_repeat_factor=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_repeat_factor = max(int(epoch_repeat_factor), 1)

    def __len__(self):
        return len(self.ids) * self.epoch_repeat_factor


def get_scalematch_dataset_cls(dataset_name):
    if dataset_name in NATURAL_IMAGE_DATASETS:
        return NaturalSemiDataset, "semi"
    if dataset_name in REMOTE_SENSING_DATASETS:
        return ScaleMatchRemoteSemiDataset, "semi_rs"
    raise ValueError(
        f"Unsupported dataset for scalematch: {dataset_name}. "
        f"Please register it in NATURAL_IMAGE_DATASETS or REMOTE_SENSING_DATASETS."
    )


def get_eval_mode(cfg):
    if "eval_mode" in cfg:
        return cfg["eval_mode"]
    if cfg["dataset"] == "cityscapes":
        return "slide_window"
    return "original"


def get_scalematch_recipe(cfg):
    return {
        "img_scales": cfg.get("img_scales", DEFAULT_IMG_SCALES),
        "feat_s_scales": cfg.get("feat_s_scales", DEFAULT_FEAT_S_SCALES),
        "feat_l_scales": cfg.get("feat_l_scales", DEFAULT_FEAT_L_SCALES),
        "conf_thresh": cfg.get(
            "conf_thresh", 0.0 if cfg["dataset"] == "cityscapes" else 0.95
        ),
        "warm_up": cfg.get("warm_up", OFFICIAL_WARM_UP),
    }


def build_scalematch_model(cfg):
    backbone_size = cfg["backbone"].split("_")[-1]
    backbone_version = cfg["backbone"].split("_")[0]
    model_name = cfg.get("model", "dpt").lower()
    model_kwargs = {**MODEL_CONFIGS[backbone_size], "nclass": cfg["nclass"]}

    if model_name != "dpt":
        raise ValueError(
            f"Unsupported ScaleMatch model '{cfg.get('model')}'. "
            "This port currently supports only 'dpt'."
        )

    model = DPT_ScaleMatch(
        **model_kwargs,
        backbone_version=backbone_version,
    )
    return model, backbone_version


def get_parser():
    parser = argparse.ArgumentParser(
        description="ScaleMatch: Multi-scale Semi-Supervised Semantic Segmentation"
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
    debug_cfg = get_debug_cfg(cfg)
    debug_enabled = debug_cfg["enabled"]
    viz = None

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        all_args.setdefault("eval_mode", get_eval_mode(cfg))
        logger.info("{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)
        if debug_enabled:
            from util.viz import Visualizer

            filename = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz = Visualizer(
                save_dir=f"./viz/{filename}_scalematch", dataset=cfg["dataset"]
            )

    cudnn.enabled = True
    cudnn.benchmark = True

    model, backbone_version = build_scalematch_model(cfg)

    backbone_ckpt_path = f'./pretrained/{cfg["backbone"]}.pth'
    if rank == 0:
        logger.info(f"Backbone version: {backbone_version}")
        logger.info(f"Backbone checkpoint: {backbone_ckpt_path}")
    state_dict = torch.load(backbone_ckpt_path, map_location="cpu")
    load_result = model.backbone.load_state_dict(state_dict)
    if rank == 0:
        logger.info(
            "Backbone load result | missing_keys=%d unexpected_keys=%d"
            % (len(load_result.missing_keys), len(load_result.unexpected_keys))
        )
        if load_result.missing_keys:
            logger.info(f"Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.info(f"Unexpected keys: {load_result.unexpected_keys}")
        logger.info(f"Loaded {backbone_version} backbone weights successfully")

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
        logger.info(
            "Optimizer: AdamW | lr_backbone=%.8f lr_head=%.8f"
            % (cfg["lr"], cfg["lr"] * cfg["lr_multi"])
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=False,
    )
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
    epoch_repeat_factor = cfg.get("epoch_repeat_factor", 1)
    if rank == 0:
        logger.info(
            f"ScaleMatch dataset loader: {dataset_loader_name} for {cfg['dataset']}"
        )
        if cfg["dataset"] in REMOTE_SENSING_DATASETS:
            logger.info(f"ScaleMatch epoch_repeat_factor={epoch_repeat_factor}")

    dataset_kwargs = {}
    if SemiDataset is ScaleMatchRemoteSemiDataset:
        dataset_kwargs["epoch_repeat_factor"] = epoch_repeat_factor
    trainset_u = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        ignore_index=ignore_index,
        **dataset_kwargs,
    )
    trainset_l = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(trainset_u.ids),
        ignore_index=ignore_index,
        **dataset_kwargs,
    )
    val_cfg = dict(cfg)
    val_cfg.setdefault("eval_mode", get_eval_mode(cfg))
    val_cfg.setdefault("ignore_index", ignore_index)
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

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(
            os.path.join(args.save_path, "latest.pth"), map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
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

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        model.train()

        log_interval = max(len(trainloader_u) // 8, 1)

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
            (img_u_w_mix, img_u_s1_mix, _, ignore_mask_mix, _, _),
        ) in enumerate(loader):
            iter_start = time.time()

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

            iters = epoch * len(trainloader_u) + i
            cutmix_img_(img_u_s1, img_u_s1_mix, cutmix_box1)

            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp):
                    pred_u_w_mix = model_noddp(img_u_w_mix, scale_factor=None)
                    if isinstance(pred_u_w_mix, dict):
                        pred_u_w_mix = pred_u_w_mix["pred_ori"]
                    conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)

                    teacher_out = model_noddp(
                        img_u_w,
                        scale_factor=random_scale,
                        feature_scale=feature_scale,
                    )
                    pred_u_w = (
                        teacher_out["pred_ori"]
                        if epoch < warm_up
                        else teacher_out["pred_joint"]
                    )
                    conf_u_w, mask_u_w = pred_u_w.detach().softmax(dim=1).max(dim=1)
            model.train()
            optimizer.zero_grad()

            mask_u_w_cutmixed1 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box1)
            conf_u_w_cutmixed1 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box1)
            ignore_mask_cutmixed1 = cutmix_mask(
                ignore_mask, ignore_mask_mix, cutmix_box1
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

            if (
                debug_enabled
                and rank == 0
                and (i % debug_cfg["grad_stats_interval"] == 0)
            ):
                writer.add_scalar(
                    "debug/grad/scale_attn", grad_norm(model_noddp.scale_attn), iters
                )
                writer.add_scalar(
                    "debug/grad/se_block", grad_norm(model_noddp.se_block), iters
                )
                writer.add_scalar(
                    "debug/grad/rwkv_layers", grad_norm(model_noddp.rwkv_layers), iters
                )
                writer.add_scalar(
                    "debug/grad/head", grad_norm(model_noddp.head), iters
                )

            scaler.step(optimizer)
            scaler.update()

            valid_mask = ignore_mask != ignore_index
            mask_ratio = (
                (conf_u_w >= conf_thresh) & valid_mask
            ).sum().float() / valid_mask.sum().clamp(min=1.0)

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

            if (
                debug_enabled
                and rank == 0
                and (i % debug_cfg["class_stats_interval"] == 0)
            ):
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
                    writer.add_scalar(
                        f"debug/{metric_name}", debug_metrics[metric_name], iters
                    )
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

            if (
                debug_enabled
                and rank == 0
                and viz is not None
                and i < debug_cfg["viz_train_iters"]
            ):
                viz.push(
                    {
                        "img_x": (img_x[0], viz.TENSOR),
                        "mask_x": (mask_x[0], viz.SEGMENTATION),
                        "pred_x_ori": (pred_x_ori.argmax(dim=1)[0], viz.SEGMENTATION),
                        "pred_x_joint": (
                            pred_x_joint.argmax(dim=1)[0],
                            viz.SEGMENTATION,
                        ),
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
                        "mask_cutmix": (
                            mask_u_w_cutmixed1[0],
                            viz.SEGMENTATION,
                        ),
                        "pred_u_s": (pred_u_s.argmax(dim=1)[0], viz.SEGMENTATION),
                        "pred_u_w_scale": (
                            pred_u_w_scale.argmax(dim=1)[0],
                            viz.SEGMENTATION,
                        ),
                        "pred_u_w_fp": (
                            pred_u_w_fp.argmax(dim=1)[0],
                            viz.SEGMENTATION,
                        ),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if (i % log_interval == 0) and (rank == 0):
                for k, v in log_avg.avgs.items():
                    writer.add_scalar(
                        "train/" + k,
                        v.item() if torch.is_tensor(v) else v,
                        iters,
                    )
                logger.info(f"Iters: {i}, " + str(log_avg))
                log_avg.reset()

        eval_mode = get_eval_mode(cfg)
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
                    "eval/%s_IoU" % CLASSES[cfg["dataset"]][i], iou, epoch
                )

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
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))

        eta_seconds = (total_epochs - (epoch + 1)) * (time.time() - start_time)


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
