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
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None
import yaml

from dataset.semi_rs import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.scalematch import ScaleMatchModel
from model.semseg.upernet import UperNet
from scalematch import (
    DEFAULT_FEAT_L_SCALES,
    DEFAULT_FEAT_S_SCALES,
    DEFAULT_IMG_SCALES,
    OFFICIAL_CONF_THRESH,
    OFFICIAL_USE_AMP,
    OFFICIAL_WARM_UP,
)
from unimatchv2_peft import apply_peft, resolve_peft_cfg, show_trainable_parameters
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


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


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


def build_writer(save_path):
    if SummaryWriter is None:
        return NullWriter()
    return SummaryWriter(save_path)


def build_scalematch_model(cfg, peft_cfg):
    model_kwargs = get_model_kwargs(cfg)
    _, backbone_version = get_backbone_info(cfg)

    if cfg["model"] == "dpt":
        base_model = DPT(**model_kwargs, backbone_version=backbone_version)
    elif cfg["model"] == "upernet":
        base_model = UperNet(**model_kwargs, backbone_version=backbone_version)
    else:
        raise ValueError(f"Unsupported model type: {cfg['model']}")

    load_result = load_backbone_checkpoint(base_model, cfg)

    if peft_cfg.get("freeze_backbone", True):
        if hasattr(base_model, "lock_backbone"):
            base_model.lock_backbone()
        else:
            for param in base_model.backbone.parameters():
                param.requires_grad = False

    base_model = apply_peft(base_model, peft_cfg, cfg)
    model = ScaleMatchModel(base_model, cfg["nclass"])
    return model, load_result


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


def unpack_unlabeled_batch(batch):
    if len(batch) == 6:
        return (*batch, None)
    if len(batch) == 7:
        return batch
    raise ValueError(f"Unexpected unlabeled batch size: {len(batch)}")


def sample_synced_choice(options, device, rank):
    if len(options) == 1:
        return options[0]
    index = torch.zeros(1, device=device, dtype=torch.long)
    if rank == 0:
        index.fill_(random.randrange(len(options)))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(index, src=0)
    return options[index.item()]


def main(args, cfg):
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    peft_cfg = resolve_peft_cfg(cfg, args)
    ignore_index = cfg.get("ignore_index", 255)
    amp = OFFICIAL_USE_AMP
    img_scales = DEFAULT_IMG_SCALES
    feat_s_scales = DEFAULT_FEAT_S_SCALES
    feat_l_scales = DEFAULT_FEAT_L_SCALES
    warm_up = OFFICIAL_WARM_UP
    conf_thresh = OFFICIAL_CONF_THRESH

    if rank == 0:
        all_args = {
            **cfg,
            **vars(args),
            "ngpus": world_size,
            "scalematch_img_scales": img_scales,
            "scalematch_feat_s_scales": feat_s_scales,
            "scalematch_feat_l_scales": feat_l_scales,
            "scalematch_warm_up": warm_up,
            "scalematch_conf_thresh": conf_thresh,
            "scalematch_amp": amp,
        }
        logger.info("{}\n".format(pprint.pformat(all_args)))
        logger.info(
            "Running ScaleMatch + PEFT with method=%s, target_modules=%s, freeze_backbone=%s",
            peft_cfg["method"],
            peft_cfg["target_modules"],
            peft_cfg["freeze_backbone"],
        )
        os.makedirs(args.save_path, exist_ok=True)
        writer = build_writer(args.save_path)
    else:
        writer = NullWriter()

    cudnn.enabled = True
    cudnn.benchmark = True

    model, load_result = build_scalematch_model(cfg, peft_cfg)
    optimizer = build_optimizer(model, cfg)

    if rank == 0:
        logger.info("Total params: %.1fM", count_params(model))
        logger.info("Encoder params: %.1fM", count_params(model.backbone))
        logger.info("Decoder params: %.1fM", count_params(model.head))
        logger.info(
            "backbone load_result missing_keys=%s unexpected_keys=%s",
            list(getattr(load_result, "missing_keys", [])),
            list(getattr(load_result, "unexpected_keys", [])),
        )
        show_trainable_parameters(model, logger)

    local_rank = get_local_rank()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    log_cuda_memory(
        logger,
        rank,
        "after_model_to_cuda",
        local_rank=local_rank,
        save_path=args.save_path,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )
    log_cuda_memory(
        logger,
        rank,
        "after_ddp_wrap",
        local_rank=local_rank,
        save_path=args.save_path,
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

    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)

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
        cfg["dataset"], cfg["data_root"], "val", ignore_value=ignore_index
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

    trainsampler_u_mix = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u_mix = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_u_mix,
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
    previous_best = 0.0
    best_epoch = -1
    epoch = -1
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
        best_epoch = checkpoint.get("best_epoch", -1)

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
        trainloader_u_mix.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)
        model.train()

        for i, ((img_x, mask_x), batch_u, batch_u_mix) in enumerate(loader):
            (
                img_u_w,
                img_u_s1,
                img_u_s2,
                ignore_mask,
                cutmix_box1,
                cutmix_box2,
                _,
            ) = unpack_unlabeled_batch(batch_u)
            (
                img_u_w_mix,
                img_u_s1_mix,
                img_u_s2_mix,
                ignore_mask_mix,
                _,
                _,
                _,
            ) = unpack_unlabeled_batch(batch_u_mix)

            img_x = img_x.cuda(local_rank, non_blocking=True)
            mask_x = mask_x.cuda(local_rank, non_blocking=True)
            img_u_w = img_u_w.cuda(local_rank, non_blocking=True)
            img_u_s1 = img_u_s1.cuda(local_rank, non_blocking=True)
            img_u_s2 = img_u_s2.cuda(local_rank, non_blocking=True)
            ignore_mask = ignore_mask.cuda(local_rank, non_blocking=True)
            cutmix_box1 = cutmix_box1.cuda(local_rank, non_blocking=True)
            cutmix_box2 = cutmix_box2.cuda(local_rank, non_blocking=True)
            img_u_w_mix = img_u_w_mix.cuda(local_rank, non_blocking=True)
            img_u_s1_mix = img_u_s1_mix.cuda(local_rank, non_blocking=True)
            img_u_s2_mix = img_u_s2_mix.cuda(local_rank, non_blocking=True)
            ignore_mask_mix = ignore_mask_mix.cuda(local_rank, non_blocking=True)

            cutmix_img_(img_u_s1, img_u_s1_mix, cutmix_box1)
            cutmix_img_(img_u_s2, img_u_s2_mix, cutmix_box2)

            random_scale = sample_synced_choice(img_scales, img_x.device, rank)
            feature_scale = sample_synced_choice(
                feat_s_scales if random_scale > 1 else feat_l_scales,
                img_x.device,
                rank,
            )

            with torch.cuda.amp.autocast(enabled=amp):
                inference_model = model.module if hasattr(model, "module") else model
                inference_model.eval()
                with torch.no_grad():
                    pred_u_w_mix = inference_model(
                        img_u_w_mix, scale_factor=None, scales=None
                    )
                    conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)

                model.train()

                num_lb = img_x.shape[0]
                pred = model(
                    torch.cat((img_x, img_u_w)),
                    scale_factor=random_scale,
                    feature_scale=feature_scale,
                    plain_inputs=img_u_s1,
                )
                pred_u_s = pred["pred_plain"]

                if epoch < warm_up:
                    pred_u_w = pred["pred_ori"][num_lb:]
                else:
                    pred_u_w = pred["pred_joint"][num_lb:]

                pred_u_w = pred_u_w.detach()
                conf_u_w, mask_u_w = pred_u_w.softmax(dim=1).max(dim=1)

                mask_u_w_cutmixed1 = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box1)
                conf_u_w_cutmixed1 = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box1)
                ignore_mask_cutmixed1 = cutmix_mask(
                    ignore_mask, ignore_mask_mix, cutmix_box1
                )

                pred_x_joint = pred["pred_joint"][:num_lb]
                pred_u_w_scale = pred["pred_size"][num_lb:]
                pred_u_w_fp = pred["pred_fp"][num_lb:]

                loss_x = criterion_l(pred_x_joint, mask_x)

                loss_u_s1 = criterion_u(pred_u_s, mask_u_w_cutmixed1)
                loss_u_s1 = confidence_weighted_loss(
                    loss_u_s1,
                    conf_u_w_cutmixed1,
                    ignore_mask_cutmixed1,
                    ignore_index=ignore_index,
                    conf_thresh=conf_thresh,
                )
                loss_u_size = criterion_u(pred_u_w_scale, mask_u_w)
                loss_u_size = confidence_weighted_loss(
                    loss_u_size,
                    conf_u_w,
                    ignore_mask,
                    ignore_index=ignore_index,
                    conf_thresh=conf_thresh,
                )
                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = confidence_weighted_loss(
                    loss_u_w_fp,
                    conf_u_w,
                    ignore_mask,
                    ignore_index=ignore_index,
                    conf_thresh=conf_thresh,
                )

                loss_standard = (
                    loss_u_s1 * 0.25 + loss_u_size * 0.25 + loss_u_w_fp * 0.5
                )
                total_loss = (loss_x + loss_standard) / 2.0

            optimizer.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            valid_mask = ignore_mask != ignore_index
            mask_ratio = (
                ((conf_u_w >= conf_thresh) & valid_mask).sum().item()
                / valid_mask.sum().clamp(min=1).item()
            )
            log_avg.update(
                {
                    "Total loss": total_loss,
                    "Loss x": loss_x,
                    "Loss u_s": loss_u_s1,
                    "Loss u_scale": loss_u_size,
                    "Loss w_fp_scale": loss_u_w_fp,
                    "Mask ratio": mask_ratio,
                }
            )

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                for key, value in log_avg.avgs.items():
                    writer.add_scalar(
                        "train/" + key,
                        value.item() if torch.is_tensor(value) else value,
                        iters,
                    )

            if (i % max(1, len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info("Iters: %s, %s", i, log_avg)
                log_avg.reset()

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode", "slide_window" if cfg["dataset"] == "cityscapes" else "original"
        )
        val_cfg.setdefault("ignore_index", ignore_index)
        eval_mode = val_cfg["eval_mode"]
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
                eval_mode,
                mIoU,
            )
            writer.add_scalar("eval/mIoU", mIoU, epoch)
            for cls_idx, iou in enumerate(iou_class):
                writer.add_scalar(
                    "eval/%s_IoU" % (CLASSES[cfg["dataset"]][cls_idx]),
                    iou,
                    epoch,
                )

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
    cfg.setdefault("ignore_index", 255)
    main(args, cfg)
