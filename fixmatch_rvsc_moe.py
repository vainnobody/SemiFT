import os
import time
import yaml
import pprint
import logging
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataset.semi_rvs import SemiDataset
from dataset.val import ValDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.train_utils import (
    DictAverageMeter,
    confidence_weighted_loss,
    cutmix_img_,
    cutmix_mask,
)
from supervised import validation_cpu

# from model.semseg.dpt import DPT
# from model.semseg.upernet_dinov2 import UperNet
from model.semseg.upernet_single import UperNet
# from model.backbone.dinov2 import DINOv2

import torchvision.transforms.functional as TF
import torch.nn.functional as F

from peft.tuners.semift import SemiFTConfig, AdaptModel


parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--backbone", type=str, default="swint", required=True)
parser.add_argument("--init_backbone", type=str, default="imp", required=True)
parser.add_argument("--decoder", type=str, default="upernet", required=True)
parser.add_argument("--labeled-id-path", type=str, required=True)
parser.add_argument("--unlabeled-id-path", type=str, required=True)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--load", type=str, default="backbone")
parser.add_argument("--interval", type=int, default=1)


def set_seeds(seed=2024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scale_back(pred_c_back, mask_c_back, size, box):
    B, C, h, w = pred_c_back.shape

    preds = []
    masks = []
    mask_s = []

    for i in range(B):
        x, y, x_c, y_c, s, theta = box[i]
        x, y, x_c, y_c, s, theta = (
            x.item(),
            y.item(),
            x_c.item(),
            y_c.item(),
            s.item(),
            theta.item(),
        )

        pred = TF.rotate(
            pred_c_back[i], angle=-theta, interpolation=TF.InterpolationMode.BILINEAR
        )
        mask = TF.rotate(
            mask_c_back[i], angle=-theta, interpolation=TF.InterpolationMode.NEAREST
        )
        aligned_ctx = torch.zeros((C, h, w)).to(pred_c_back.device)
        aligned_mask = torch.zeros((1, h, w)).to(pred_c_back.device)

        rect_m = [x, y, x + size, y + size]
        rect_s = [x_c, y_c, x_c + size * s, y_c + size * s]

        inter_x1 = max(rect_m[0], rect_s[0])
        inter_y1 = max(rect_m[1], rect_s[1])
        inter_x2 = min(rect_m[2], rect_s[2])
        inter_y2 = min(rect_m[3], rect_s[3])

        m_x1 = int(round(inter_x1 - x))
        m_y1 = int(round(inter_y1 - y))
        m_x2 = int(round(inter_x2 - x))
        m_y2 = int(round(inter_y2 - y))

        target_h = m_y2 - m_y1
        target_w = m_x2 - m_x1

        c_x1 = int(round((inter_x1 - x_c) / s))
        c_y1 = int(round((inter_y1 - y_c) / s))
        c_x2 = int(round((inter_x2 - x_c) / s))
        c_y2 = int(round((inter_y2 - y_c) / s))

        c_x1, c_x2 = max(0, c_x1), min(size, c_x2)
        c_y1, c_y2 = max(0, c_y1), min(size, c_y2)

        pred_patch = pred[:, c_y1:c_y2, c_x1:c_x2]
        mask_patch = mask[:, c_y1:c_y2, c_x1:c_x2]

        pred_patch = F.interpolate(
            pred_patch.unsqueeze(0), size=(target_h, target_w), mode="bilinear"
        ).squeeze(0)

        mask_patch = F.interpolate(
            mask_patch.unsqueeze(0), size=(target_h, target_w), mode="nearest"
        ).squeeze(0)

        # print(mask_patch.shape)
        aligned_ctx[:, m_y1:m_y2, m_x1:m_x2] = pred_patch
        aligned_mask[:, m_y1:m_y2, m_x1:m_x2] = mask_patch

        preds.append(aligned_ctx)
        masks.append(aligned_mask)

    return torch.stack(preds, dim=0), torch.stack(masks, dim=0)


def apply_peft(model, cfg):
    peft_config = SemiFTConfig(
        method=cfg['peft'],
        target_modules=cfg['target_modules'],
        modules_to_save='head',
        bias='lora_only',
        nclass = cfg['nclass'],
    )

    model = AdaptModel(peft_config, model)
    return model

def show_trainable_parameters(model, logger):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    trainable_params_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_params_names.append(name)
    
    percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    
    logger.info('--- 模型可训练参数 ---')
    logger.info('--- 可训练模块/参数列表---')

    for name in trainable_params_names:
        logger.info(f' - {name}')
        
    logger.info("\n --- 统计信息 ---")
    logger.info(f" - 总参数数量: {total_params:,}")
    logger.info(f" - 可训练参数数量: {trainable_params:,}")
    logger.info(f" - 可训练参数占比: {percentage:.2f}%")


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg['peft'] = 'semift'
    cfg['target_modules'] = ['mlp']

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    ddp = True if world_size > 1 else False
    amp = cfg["amp"]

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    conf_thresh = cfg["conf_thresh"]
    # model_configs = {
    #     'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #     'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #     'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #     'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    # }
    # backbone = DINOv2(model_name='base')
    # state_dict = torch.load(f'/data1/users/lvliang/project_123/S5_finetune/pretrained/{cfg["backbone"]}.pth')
    # backbone.load_state_dict(state_dict)
    # model = UperNet(args, cfg, backbone)
    model = UperNet(args, cfg)

    #lock backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    
    model = apply_peft(model, cfg)
    logger.info(model)
    show_trainable_parameters(model, logger)
    # model = DeepLabV3Plus(cfg)
    # model = DeepLabV3Plus(cfg)

    # model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
``
    from mmengine.optim import build_optim_wrapper

    optim_wrapper = dict(
        optimizer=dict(
            type="AdamW", lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=0.01
        ),
        paramwise_cfg=dict(
            custom_keys={
                "absolute_pos_embed": dict(decay_mult=0.0),
                "relative_position_bias_table": dict(decay_mult=0.0),
                "norm": dict(decay_mult=0.0),
            }
        ),
    )

    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.optimizer, cfg["epochs"], eta_min=0, last_epoch=-1
    )

    if rank == 0:
        logger.info("Total params: {:.1f}M\n".format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            output_device=local_rank,
            find_unused_parameters=True,
        )

    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(
            local_rank
        )
    else:
        raise NotImplementedError(
            "%s criterion is not implemented" % cfg["criterion"]["name"]
        )

    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)

    trainset_u = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        size=cfg["crop_size"],
        ignore_value=cfg["ignore_index"],
        id_path=args.unlabeled_id_path,
    )

    trainset_l = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        size=cfg["crop_size"],
        ignore_value=cfg["ignore_index"],
        id_path=args.labeled_id_path,
        nsample=len(trainset_u.ids),
    )
    valset = ValDataset(cfg["dataset"], cfg["data_root"], "val")

    if ddp:
        trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
        trainloader_l = DataLoader(
            trainset_l,
            batch_size=cfg["batch_size"],
            pin_memory=True,
            num_workers=8,
            drop_last=False,
            sampler=trainsampler_l,
        )
        trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
        trainloader_u = DataLoader(
            trainset_u,
            batch_size=cfg["batch_size"],
            pin_memory=True,
            num_workers=8,
            drop_last=False,
            sampler=trainsampler_u,
        )
        valsampler = torch.utils.data.distributed.DistributedSampler(valset)
        valloader = DataLoader(
            valset,
            batch_size=4,
            pin_memory=True,
            num_workers=8,
            drop_last=False,
            sampler=valsampler,
        )
    else:
        trainloader_l = DataLoader(
            trainset_l,
            batch_size=cfg["batch_size"],
            pin_memory=True,
            num_workers=8,
            shuffle=True,
            drop_last=True,
        )
        trainloader_u = DataLoader(
            trainset_u,
            batch_size=cfg["batch_size"],
            pin_memory=True,
            num_workers=1,
            shuffle=True,
            drop_last=True,
        )
        valloader = DataLoader(
            valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False
        )

    total_epochs = cfg["epochs"]
    total_iters = len(trainloader_u) * total_epochs
    epoch = -1
    previous_best = 0.0
    ETA = 0.0
    scaler = torch.cuda.amp.GradScaler()
    is_best = False

    for epoch in range(epoch + 1, total_epochs):
        start_time = time.time()
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}, ETA: {:.2f}M".format(
                    epoch, optimizer.param_groups[0]["lr"], previous_best, ETA / 60
                )
            )

        log_avg = DictAverageMeter()

        if ddp:
            trainloader_l.sampler.set_epoch(epoch)
            trainloader_u.sampler.set_epoch(epoch)

        start_time = time.time()
        model.train()
        total_loss = AverageMeter()
        loader = zip(trainloader_l, trainloader_u)

        for i, ((img_x , mask_x, img_x_c, mask_x_c),
            (img_u_w, img_u_s, img_u_c, ignore_mask, cutmix_box, _, box, mask_c)) in enumerate(loader):

            t0 = time.time()
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_x_c, mask_x_c = img_x_c.cuda(), mask_x_c.cuda()
            img_u_w, img_u_s, img_u_c, ignore_mask = (
                img_u_w.cuda(),
                img_u_s.cuda(),
                img_u_c.cuda(),
                ignore_mask.cuda(),
            )
            mask_c = mask_c.cuda()
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            with torch.cuda.amp.autocast(enabled=amp):

                model.train()
                pred_x, pred_u_w = model(torch.cat((img_x, img_u_w))).split(
                    [num_lb, num_ulb]
                )
                pred_u_s = model(img_u_s)
                pred_x_rvs, pred_u_rvs = model(torch.cat((img_x_c, img_u_c))).split(
                    [num_lb, num_ulb]
                )
                pred_u_w = pred_u_w.detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)
                pred_u_w_rvs, mask_u_w_rvs, conf_u_w_rvs = pred_u_w.clone(), mask_u_w.clone(), conf_u_w.clone()
                
                pred_recovered, valid_masks = scale_back(
                    pred_u_rvs, mask_c, cfg['crop_size'], box,
                )
                
                valid_masks = valid_masks.squeeze(1)
                mask_u_w_rvs[valid_masks==0] = cfg['criterion']['kwargs']['ignore_index']
                
                mask_u_w, conf_u_w = mask_u_w.clone(), conf_u_w.clone()
                loss_x = criterion_l(pred_x, mask_x)
                loss_x_rvs = criterion_l(pred_x_rvs, mask_x_c)
                
                loss_u_s = criterion_u(pred_u_s, mask_u_w)
                loss_u_s = confidence_weighted_loss(
                    loss_u_s,
                    conf_u_w,
                    ignore_mask,
                    cfg["ignore_index"],
                    conf_thresh=conf_thresh,
                )
                loss_u_s_rvs = criterion_l(pred_recovered, mask_u_w_rvs)
                loss_u_s_rvs = confidence_weighted_loss(
                    loss_u_s_rvs,
                    conf_u_w,
                    mask_u_w_rvs,
                    cfg["ignore_index"],
                    conf_thresh=conf_thresh,
                )
                # aux_loss = sum(m.aux_loss for m in model.modules() if hasattr(m, 'aux_loss'))
                aux_loss = 0
                mask_ratio = (
                    (conf_u_w >= conf_thresh) & (ignore_mask != cfg["ignore_index"])
                ).sum().item() / (ignore_mask != cfg["ignore_index"]).sum()
                total_loss = (loss_x + loss_x_rvs + loss_u_s + loss_u_s_rvs) / 4.0

            if ddp:
                torch.distributed.barrier()

            optimizer.zero_grad()
            if amp:
                loss = scaler.scale(total_loss)
                # torch.autograd.set_detect_anomaly(True)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # Logging
            log_avg.update(
                {
                    "iter time": time.time() - t0,
                    "Total loss": total_loss,  # Logging original loss (=total loss), not AMP scaled loss
                    "Loss x": loss_x,
                    "Loss x_rvs": loss_x_rvs,
                    "Loss u_s": loss_u_s,
                    "Loss u_s_rvs": loss_u_s_rvs,
                    'aux_loss': aux_loss,
                    "Mask ratio": mask_ratio,
                }
            )

            iters = epoch * len(trainloader_u) + i
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(f"Iters: {i}, " + str(log_avg))
                log_avg.reset()

        if epoch % args.interval == 0:
            start_time = time.time()
            mIoU, mAcc, mF1, allAcc, iou_class, F1_class = validation_cpu(
                cfg, model, valloader
            )
            end_time = time.time()

            if rank == 0:
                for cls_idx, iou in enumerate(iou_class):
                    logger.info(
                        "***** Evaluation ***** >>>> Class [{:} {:}] "
                        "IoU: {:.2f}".format(
                            cls_idx, CLASSES[cfg["dataset"]][cls_idx], iou
                        )
                    )
                # logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
                logger.info(
                    "Last: validation epoch [{}/{}]: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}. Cost {:.4f} secs".format(
                        epoch + 1,
                        cfg["epochs"],
                        mIoU,
                        mAcc,
                        mF1,
                        allAcc,
                        end_time - start_time,
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
        scheduler.step()


if __name__ == "__main__":
    set_seeds(1234)
    main()
