import argparse
import logging
import glob
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(
    description="Fully-Supervised Training in Semantic Segmentation"
)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--labeled-id-path", type=str, required=True)
parser.add_argument("--unlabeled-id-path", type=str, default=None)
parser.add_argument("--pretrained-path", type=str, default=None)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)


# @torch.no_grad()
@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()

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
        intersection, union, target, predict = intersectionAndUnion(
            gray, target, cfg["nclass"], cfg["ignore_index"]
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        predict_meter.update(predict)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10)
    F1_class = 2 * (precise_class * accuracy_class) / (precise_class + accuracy_class)

    if cfg["dataset"] == "iSAID":
        mIoU = np.mean(iou_class[1:]) * 100.0
        mAcc = np.mean(accuracy_class[1:]) * 100.0
        mF1 = np.mean(F1_class[1:]) * 100.0
        allAcc = sum(intersection_meter.sum[1:]) / (sum(target_meter.sum[1:]) + 1e-10)
    else:
        mIoU = np.nanmean(iou_class) * 100.0
        mAcc = np.nanmean(accuracy_class) * 100.0
        mF1 = np.nanmean(F1_class) * 100.0
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc, iou_class, F1_class


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


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

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
    model = DPT(
        **{**model_configs[cfg["backbone"].split("_")[-1]], "nclass": cfg["nclass"]}
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
        logger.info("Total params: {:.1f}M\n".format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
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
    else:
        raise NotImplementedError(
            "%s criterion is not implemented" % cfg["criterion"]["name"]
        )

    n_upsampled = {"pascal": 3000, "cityscapes": 3000, "ade20k": 6000, "coco": 30000}
    trainset = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=n_upsampled[cfg["dataset"]],
    )
    valset = SemiDataset(cfg["dataset"], cfg["data_root"], "val")

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset,
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

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, LR: {:.7f}, Previous best: {:.2f}".format(
                    epoch, optimizer.param_groups[0]["lr"], previous_best
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

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss.item(), iters)

            if (i % (len(trainloader) // 8) == 0) and (rank == 0):
                logger.info("Iters: {:}, Total loss: {:.3f}".format(i, total_loss.avg))

        eval_mode = "sliding_window" if cfg["dataset"] == "cityscapes" else "original"
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [{:} {:}] "
                    "IoU: {:.2f}".format(cls_idx, CLASSES[cfg["dataset"]][cls_idx], iou)
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

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    main()
