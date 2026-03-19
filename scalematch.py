"""
ScaleMatch: Multi-scale semi-supervised semantic segmentation.

Ported from the official ScaleMatch training logic onto the SemiFT DPT backbone.
"""

import argparse
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
OFFICIAL_IMG_SCALES = [0.25, 0.5, 1.5, 2.0]
OFFICIAL_FEAT_S_SCALES = [0.75]
OFFICIAL_FEAT_L_SCALES = [1.25]
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
        "img_scales": cfg.get("img_scales", OFFICIAL_IMG_SCALES),
        "feat_s_scales": cfg.get("feat_s_scales", OFFICIAL_FEAT_S_SCALES),
        "feat_l_scales": cfg.get("feat_l_scales", OFFICIAL_FEAT_L_SCALES),
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

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        all_args.setdefault("eval_mode", get_eval_mode(cfg))
        logger.info("{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

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

            with torch.cuda.amp.autocast(enabled=amp):
                model.eval()
                with torch.no_grad():
                    pred_u_w_mix = model_noddp(
                        img_u_w_mix, scale_factor=None, scales=None
                    )
                    if isinstance(pred_u_w_mix, dict):
                        pred_u_w_mix = pred_u_w_mix["pred_ori"]
                    conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)
                model.train()

                num_lb = img_x.shape[0]
                student_out = model(
                    torch.cat((img_x, img_u_w)),
                    scale_factor=random_scale,
                    feature_scale=feature_scale,
                    strong_inputs=img_u_s1,
                    pseudo_mode="ori" if epoch < warm_up else "joint",
                )
                pred_u_s = student_out["pred_strong"]
                pred_x_joint = student_out["pred_joint"][:num_lb]
                pred_x_ori = student_out["pred_ori"][:num_lb]
                pred_u_w_scale = student_out["pred_size"][num_lb:]
                pred_u_w_fp = student_out["pred_fp"][num_lb:]
                pred_u_w = student_out["pseudo_logits"][num_lb:]
                conf_u_w, mask_u_w = pred_u_w.softmax(dim=1).max(dim=1)

                mask_u_w_cutmixed1 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box1)
                conf_u_w_cutmixed1 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box1)
                ignore_mask_cutmixed1 = cutmix_mask(
                    ignore_mask, ignore_mask_mix, cutmix_box1
                )

                loss_u_s1 = criterion_u(pred_u_s, mask_u_w_cutmixed1)
                loss_u_s1 = confidence_weighted_loss(
                    loss_u_s1,
                    conf_u_w_cutmixed1,
                    ignore_mask_cutmixed1,
                    ignore_index,
                    conf_thresh=conf_thresh,
                )

                loss_x_joint = criterion_l(pred_x_joint, mask_x)
                loss_x_ori = criterion_l(pred_x_ori, mask_x)
                loss_x = loss_x_ori if epoch < warm_up else loss_x_joint

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

                loss_standard = (
                    0.25 * loss_u_s1 + 0.25 * loss_u_size + 0.5 * loss_u_w_fp
                )
                total_loss = (loss_x + loss_standard) / 2.0

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            valid_mask = ignore_mask != ignore_index
            mask_ratio = (
                (conf_u_w >= conf_thresh) & valid_mask
            ).sum().float() / valid_mask.sum().clamp(min=1.0)

            log_avg.update(
                {
                    "iter_time": time.time() - iter_start,
                    "Total_loss": total_loss,
                    "Loss_x": loss_x,
                    "Loss_x_joint": loss_x_joint.detach(),
                    "Loss_x_ori": loss_x_ori.detach(),
                    "Loss_u_s": loss_u_s1,
                    "Loss_u_scale": loss_u_size,
                    "Loss_u_fp": loss_u_w_fp,
                    "Conf_x_joint": pred_x_joint.detach().softmax(dim=1).amax(dim=1).mean(),
                    "Conf_x_ori": pred_x_ori.detach().softmax(dim=1).amax(dim=1).mean(),
                    "Conf_u_w": conf_u_w.mean(),
                    "Mask_ratio": mask_ratio,
                    "LR_backbone": optimizer.param_groups[0]["lr"],
                    "LR_head": optimizer.param_groups[1]["lr"],
                }
            )

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
