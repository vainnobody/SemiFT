import argparse
from copy import deepcopy
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi_rs import SemiDataset
from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.my_upernet import MyUperNet
from model.semseg.upernet import UperNet
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.focal import FocalLoss
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

from util.viz import Visualizer

import numpy as np


def evaluate(model, loader, mode, cfg, multiplier=None):
    model.eval()
    assert mode in ["original", "center_crop", "sliding_window"]
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:

            img = img.cuda()

            if mode == "sliding_window":
                grid = cfg["crop_size"]
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()

                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row : row + grid, col : col + grid])
                        final[:, :, row : row + grid, col : col + grid] += pred.softmax(
                            dim=1
                        )
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)

                pred = final

            else:
                assert mode == "original"

                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    if multiplier == 512:
                        new_h, new_w = 512, 512
                    else:
                        # Ensure dimensions are multiples of multiplier (patch_size)
                        new_h, new_w = (
                            int(ori_h / multiplier + 0.5) * multiplier,
                            int(ori_w / multiplier + 0.5) * multiplier,
                        )
                    img = F.interpolate(
                        img, (new_h, new_w), mode="bilinear", align_corners=True
                    )

                pred = model(img)

                if multiplier is not None:
                    pred = F.interpolate(
                        pred, (ori_h, ori_w), mode="bilinear", align_corners=True
                    )

            pred = pred.argmax(dim=1)

            intersection, union, target = intersectionAndUnion(
                pred.cpu().numpy(), mask.numpy(), cfg["nclass"], 255
            )

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()

    for x, y, _ in valid_loader:
        x = x.cuda()
        if cfg["eval_mode"] == "slide_window":
            b, _, h, w = x.shape  # 获取输入图像的尺寸 (batch, channels, height, width)
            final = torch.zeros(b, cfg["nclass"], h, w).cuda()  # 用于存储最终预测结果
            size = cfg["crop_size"]
            step = 510
            b = 0
            a = 0
            while a <= int(h / step):
                while b <= int(w / step):
                    sub_input = x[
                        :,
                        :,
                        min(a * step, h - size) : min(a * step + size, h),
                        min(b * step, w - size) : min(b * step + size, w),
                    ]
                    # print("sub_input.shape", sub_input.shape)
                    mask = model(sub_input)
                    final[
                        :,
                        :,
                        min(a * step, h - size) : min(a * step + size, h),
                        min(b * step, w - size) : min(b * step + size, w),
                    ] += mask
                    b += 1
                b = 0
                a += 1
            o = final.argmax(dim=1)

        elif cfg["eval_mode"] == "resize":
            # 使用缩放方式进行预测
            original_shape = x.shape[-2:]  # 保存原始图像的尺寸 (h, w)
            resized_x = F.interpolate(
                x, size=cfg["crop_size"], mode="bilinear", align_corners=True
            )
            resized_o = model(resized_x)
            # 将预测结果复原到原始尺寸
            o = F.interpolate(
                resized_o, size=original_shape, mode="bilinear", align_corners=True
            )
            o = o.argmax(dim=1)

        else:
            # 直接进行预测（非滑动窗口模式）

            o = model(x)
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


def get_parser():
    parser = argparse.ArgumentParser(
        description="Reproduced FixMatch with an EMA Teacher for Semi-Supervised Semantic Segmentation"
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

    cfg["batch_size"] *= 2

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

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
    # DINOv2 uses patch_size=14, DINOv3 uses patch_size=16
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

    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)

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

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )

    if cfg["criterion"]["name"] == "CELoss":
        criterion = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(
            local_rank
        )
    elif cfg["criterion"]["name"] == "FocalLoss":
        criterion = FocalLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    else:
        raise NotImplementedError(
            "%s criterion is not implemented" % cfg["criterion"]["name"]
        )

    trainset_l = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        ignore_index=cfg["ignore_index"],
    )
    valset = ValDataset(
        cfg["dataset"], cfg["data_root"], "val", ignore_value=cfg["ignore_index"]
    )

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler,
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

    iters = 0
    total_iters = len(trainloader) * cfg["epochs"]
    previous_best = 0.0
    best_epoch = -1
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(
            os.path.join(args.save_path, "latest.pth"), map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    from datetime import datetime

    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz = Visualizer(save_dir=f"./viz/{filename}", dataset=cfg["dataset"])

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}".format(
                    epoch, previous_best, best_epoch
                )
            )

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()
            pred = model(img)
            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i < 10:
                viz.push(
                    {
                        "img": (img[0], Visualizer.TENSOR),
                        "mask": (mask[0], Visualizer.SEGMENTATION),
                        "pred": (pred.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)

            if (i % (len(trainloader) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total loss: {:.3f}".format(
                        i,
                        optimizer.param_groups[0]["lr"],
                        total_loss.avg,
                    )
                )

        eval_mode = "sliding_window" if cfg["dataset"] == "cityscapes" else "original"
        mIoU, iou_class = validation_cpu(cfg, model, valloader)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}".format(
                        cls_idx,
                        CLASSES[cfg["dataset"]][cls_idx],
                        iou,
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
                    "eval/%s_IoU" % (CLASSES[cfg["dataset"]][i]), iou, epoch
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
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
