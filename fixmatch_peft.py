import argparse
from copy import deepcopy
import logging
import os
import pprint

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
import yaml

from util.utils import count_params, init_log, AverageMeter, intersectionAndUnion
from util.dist_helper import setup_distributed
from util.ssl_method_utils import get_local_rank, load_checkpoint_on_cpu, save_checkpoint_to_disk, log_cuda_memory, load_backbone_checkpoint
from unimatchv2_peft import apply_peft, resolve_peft_cfg, show_trainable_parameters


def get_parser():
    parser = argparse.ArgumentParser(
        description="FixMatch + configurable PEFT for Semi-Supervised Semantic Segmentation"
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




def extract_validation_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        if "out" in output:
            return output["out"]
        raise TypeError(f"Unsupported validation output dict keys: {list(output.keys())}")
    if isinstance(output, (list, tuple)):
        if not output:
            raise TypeError("Validation output list/tuple is empty.")
        return output[0]
    raise TypeError(f"Unsupported validation output type: {type(output)!r}")


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()

    for x, y, _ in valid_loader:
        x = x.cuda()
        if cfg["eval_mode"] == "slide_window":
            batch_size, _, h, w = x.shape
            final = torch.zeros(batch_size, cfg["nclass"], h, w, device=x.device)
            count = torch.zeros(batch_size, 1, h, w, device=x.device)

            crop_size = cfg["crop_size"]
            if isinstance(crop_size, int):
                crop_h = crop_w = crop_size
            else:
                crop_h, crop_w = crop_size

            step = cfg.get("stride", 510)
            if isinstance(step, int):
                step_h = step_w = step
            else:
                step_h, step_w = step

            h_starts = list(range(0, max(h - crop_h, 0) + 1, step_h))
            w_starts = list(range(0, max(w - crop_w, 0) + 1, step_w))
            if not h_starts:
                h_starts = [0]
            if not w_starts:
                w_starts = [0]
            last_h = max(h - crop_h, 0)
            last_w = max(w - crop_w, 0)
            if h_starts[-1] != last_h:
                h_starts.append(last_h)
            if w_starts[-1] != last_w:
                w_starts.append(last_w)

            for hs in h_starts:
                for ws in w_starts:
                    he = min(hs + crop_h, h)
                    we = min(ws + crop_w, w)
                    sub_input = x[:, :, hs:he, ws:we]
                    mask = extract_validation_logits(model(sub_input))
                    final[:, :, hs:he, ws:we] += mask
                    count[:, :, hs:he, ws:we] += 1

            final = final / count.clamp_min(1.0)
            o = final.argmax(dim=1)

        elif cfg["eval_mode"] == "resize":
            original_shape = x.shape[-2:]
            resized_x = F.interpolate(
                x, size=cfg["crop_size"], mode="bilinear", align_corners=True
            )
            resized_o = extract_validation_logits(model(resized_x))
            o = F.interpolate(
                resized_o, size=original_shape, mode="bilinear", align_corners=True
            )
            o = o.argmax(dim=1)

        else:
            o = extract_validation_logits(model(x))
            o = o.max(1)[1]

        gray = np.uint8(o.cpu().numpy())
        target = np.array(y, dtype=np.int32)
        intersection, union, target_area = intersectionAndUnion(
            gray, target, cfg["nclass"], cfg["ignore_index"]
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target_area)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    if cfg["dataset"] == "iSAID":
        mIoU = np.mean(iou_class[1:]) * 100.0
    else:
        mIoU = np.nanmean(iou_class) * 100.0

    return mIoU, iou_class

def build_model(cfg, peft_cfg):
    from model.semseg.dpt import DPT
    from model.semseg.upernet import UperNet

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
    patch_size = 14 if backbone_version == "dinov2" else 16

    if cfg["model"] == "dpt":
        model = DPT(
            **{**model_configs[backbone_size], "nclass": cfg["nclass"]},
            backbone_version=backbone_version,
        )
    elif cfg["model"] == "upernet":
        model = UperNet(
            **{**model_configs[backbone_size], "nclass": cfg["nclass"]},
            backbone_version=backbone_version,
        )
    else:
        raise NotImplementedError(f"Unsupported model type: {cfg['model']}")

    load_backbone_checkpoint(model, cfg)

    if peft_cfg.get("freeze_backbone", True):
        if hasattr(model, "lock_backbone"):
            model.lock_backbone()
        else:
            for p in model.backbone.parameters():
                p.requires_grad = False

    model = apply_peft(model, peft_cfg, cfg)
    return model, patch_size

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
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from dataset.semi_rs import SemiDataset
    from dataset.val import ValDataset
    from util.classes import CLASSES
    from util.ohem import ProbOhemCrossEntropy2d
    from util.focal import FocalLoss
    from util.viz import Visualizer

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    peft_cfg = resolve_peft_cfg(cfg, args)

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))
        logger.info(
            "Running FixMatch + PEFT with method=%s, target_modules=%s, freeze_backbone=%s",
            peft_cfg["method"],
            peft_cfg["target_modules"],
            peft_cfg["freeze_backbone"],
        )
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model, patch_size = build_model(cfg, peft_cfg)
    optimizer = build_optimizer(model, cfg)

    if rank == 0:
        logger.info("Total params: {:.1f}M".format(count_params(model)))
        logger.info("Encoder params: {:.1f}M".format(count_params(model.backbone)))
        logger.info("Decoder params: {:.1f}M\n".format(count_params(model.head)))
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

    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(local_rank)
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

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1

    latest_path = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(latest_path):
        log_cuda_memory(logger, rank, "before_resume_load", save_path=args.save_path)
        checkpoint = load_checkpoint_on_cpu(latest_path)
        model.load_state_dict(checkpoint["model"])
        model_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        log_cuda_memory(logger, rank, "after_resume_load", save_path=args.save_path)
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        previous_best_ema = checkpoint["previous_best_ema"]
        best_epoch = checkpoint["best_epoch"]
        best_epoch_ema = checkpoint["best_epoch_ema"]

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    from datetime import datetime

    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz = Visualizer(save_dir=f"./viz/{filename}", dataset=cfg["dataset"])

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, "
                "EMA: {:.2f} @epoch-{:}".format(
                    epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema
                )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        loader = zip(trainloader_l, trainloader_u)
        model.train()

        for i, ((img_x, mask_x), (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _)) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()

            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = img_u_s.flip(0)[
                cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1
            ]

            num_lb, num_ulb = img_x.shape[0], img_u_s.shape[0]
            pred_x, pred_u_s = model(torch.cat((img_x, img_u_s))).split([num_lb, num_ulb])

            mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = (
                mask_u_w.clone(),
                conf_u_w.clone(),
                ignore_mask.clone(),
            )

            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w.flip(0)[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w.flip(0)[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask.flip(0)[cutmix_box == 1]

            loss_x = criterion_l(pred_x, mask_x)
            loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed)
            loss_mask = (conf_u_w_cutmixed >= cfg["conf_thresh"]) & (ignore_mask_cutmixed != 255)
            loss_u_s = (loss_u_s * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = (loss_x + loss_u_s) / 2.0

            torch.distributed.barrier()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i < 10:
                viz.push(
                    {
                        "img_x": (img_x[0], Visualizer.TENSOR),
                        "mask_x": (mask_x[0], Visualizer.SEGMENTATION),
                        "pred_x": (pred_x.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                        "img_u_s": (img_u_s[0], Visualizer.TENSOR),
                        "mask_u_w_cutmixed": (mask_u_w_cutmixed[0], Visualizer.SEGMENTATION),
                        "pred_u_s": (pred_u_s.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = ((conf_u_w >= cfg["conf_thresh"]) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_s", loss_u_s.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            if (i % max(1, len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_mask_ratio.avg,
                    )
                )

        eval_mode = "sliding_window" if cfg["dataset"] == "cityscapes" else "original"
        mIoU, iou_class = validation_cpu(cfg, model, valloader)
        mIoU_ema, iou_class_ema = validation_cpu(cfg, model_ema, valloader)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, EMA: {:.2f}".format(
                        cls_idx,
                        CLASSES[cfg["dataset"]][cls_idx],
                        iou,
                        iou_class_ema[cls_idx],
                    )
                )
            logger.info(
                "***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n".format(
                    eval_mode, mIoU, mIoU_ema
                )
            )

            writer.add_scalar("eval/mIoU", mIoU, epoch)
            writer.add_scalar("eval/mIoU_ema", mIoU_ema, epoch)
            for j, iou in enumerate(iou_class):
                writer.add_scalar("eval/%s_IoU" % (CLASSES[cfg["dataset"]][j]), iou, epoch)
                writer.add_scalar(
                    "eval/%s_IoU_ema" % (CLASSES[cfg["dataset"]][j]),
                    iou_class_ema[j],
                    epoch,
                )

        is_best = mIoU >= previous_best
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch

        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
                "previous_best_ema": previous_best_ema,
                "best_epoch": best_epoch,
                "best_epoch_ema": best_epoch_ema,
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
