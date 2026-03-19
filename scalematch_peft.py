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

from dataset.val import ValDataset
from model.semseg.dpt_scalematch import DPT_ScaleMatch
from scalematch import (
    NATURAL_IMAGE_DATASETS,
    REMOTE_SENSING_DATASETS,
    ScaleMatchRemoteSemiDataset,
    get_eval_mode,
    get_scalematch_recipe,
    get_scalematch_dataset_cls,
)
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
from unimatchv2_peft import (
    DEFAULT_PEFT_CFG,
    METHOD_DEFAULT_TARGETS,
    _normalize_target_modules,
    build_peft_config,
    show_trainable_parameters,
)


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


def resolve_peft_cfg(cfg, args):
    raw_yaml_peft = cfg.get("peft", {})
    peft_cfg = dict(DEFAULT_PEFT_CFG)
    peft_cfg.update(raw_yaml_peft)

    if args.peft_method is not None:
        peft_cfg["method"] = args.peft_method
    if args.peft_target_modules is not None:
        peft_cfg["target_modules"] = args.peft_target_modules
    if args.freeze_backbone is not None:
        peft_cfg["freeze_backbone"] = args.freeze_backbone

    peft_cfg["method"] = str(peft_cfg["method"]).lower()

    modules_to_save = peft_cfg.get("modules_to_save")
    if isinstance(modules_to_save, str):
        peft_cfg["modules_to_save"] = [modules_to_save]
    elif modules_to_save is None:
        peft_cfg["modules_to_save"] = list(DEFAULT_PEFT_CFG["modules_to_save"])

    supported_methods = set(METHOD_DEFAULT_TARGETS)
    if peft_cfg["method"] not in supported_methods:
        raise ValueError(
            f"Unsupported PEFT method '{peft_cfg['method']}'. Expected one of {sorted(supported_methods)}."
        )

    target_modules_from_user = (
        args.peft_target_modules is not None or "target_modules" in raw_yaml_peft
    )
    if target_modules_from_user:
        peft_cfg["target_modules"] = _normalize_target_modules(
            peft_cfg.get("target_modules")
        )
    else:
        peft_cfg["target_modules"] = list(METHOD_DEFAULT_TARGETS[peft_cfg["method"]])

    cfg["peft"] = peft_cfg
    return peft_cfg


def apply_peft(model, peft_cfg, cfg):
    from peft.tuners.semift import AdaptModel

    return AdaptModel(build_peft_config(peft_cfg, cfg), model)


def build_model(cfg, peft_cfg):
    model_configs = {
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

    backbone_size = cfg["backbone"].split("_")[-1]
    backbone_version = cfg["backbone"].split("_")[0]
    model = DPT_ScaleMatch(
        **{**model_configs[backbone_size], "nclass": cfg["nclass"]},
        backbone_version=backbone_version,
    )

    backbone_ckpt_path = f'./pretrained/{cfg["backbone"]}.pth'
    state_dict = torch.load(backbone_ckpt_path, map_location="cpu")
    load_result = model.backbone.load_state_dict(state_dict, strict=False)

    if peft_cfg.get("freeze_backbone", True):
        model.lock_backbone()

    model = apply_peft(model, peft_cfg, cfg)
    return model, load_result, backbone_ckpt_path


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


def main(args, cfg):
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    amp = cfg.get("amp", False)
    peft_cfg = resolve_peft_cfg(cfg, args)

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        all_args.setdefault("eval_mode", get_eval_mode(cfg))
        logger.info("{}\n".format(pprint.pformat(all_args)))
        logger.info(
            "Running ScaleMatch + PEFT with method=%s, target_modules=%s, freeze_backbone=%s",
            peft_cfg["method"],
            peft_cfg["target_modules"],
            peft_cfg["freeze_backbone"],
        )
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model, load_result, backbone_ckpt_path = build_model(cfg, peft_cfg)
    optimizer = build_optimizer(model, cfg)

    if rank == 0:
        logger.info(f"Backbone checkpoint: {backbone_ckpt_path}")
        logger.info(
            "Backbone load result | missing_keys=%d unexpected_keys=%d"
            % (len(load_result.missing_keys), len(load_result.unexpected_keys))
        )
        logger.info("Total params: {:.1f}M".format(count_params(model)))
        logger.info("Encoder params: {:.1f}M".format(count_params(model.backbone)))
        logger.info("Decoder params: {:.1f}M\n".format(count_params(model.head)))
        show_trainable_parameters(model, logger)

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
        cfg["dataset"], cfg["data_root"], "val", ignore_value=ignore_index
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
                    pred_u_w_mix = model(img_u_w_mix, scale_factor=None, scales=None)
                    if isinstance(pred_u_w_mix, dict):
                        pred_u_w_mix = pred_u_w_mix["pred_ori"]
                    conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)
                model.train()

                num_lb = img_x.shape[0]
                pred = model(
                    torch.cat((img_x, img_u_w)),
                    scale_factor=random_scale,
                    feature_scale=feature_scale,
                )
                pred_u_s = model(img_u_s1, scale_factor=None, scales=None)
                if isinstance(pred_u_s, dict):
                    pred_u_s = pred_u_s["pred_ori"]

                pred_u_w = (
                    pred["pred_ori"][num_lb:]
                    if epoch < warm_up
                    else pred["pred_joint"][num_lb:]
                )
                pred_u_w = pred_u_w.detach()
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

                pred_x_joint = pred["pred_joint"][:num_lb]
                pred_u_w_scale = pred["pred_size"][num_lb:]
                pred_u_w_fp = pred["pred_fp"][num_lb:]

                loss_x = criterion_l(pred_x_joint, mask_x)
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

            if world_size > 1:
                torch.distributed.barrier()

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
                    "Loss_u_s": loss_u_s1,
                    "Loss_u_scale": loss_u_size,
                    "Loss_u_fp": loss_u_w_fp,
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
                        "train/" + k, v.item() if torch.is_tensor(v) else v, iters
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
