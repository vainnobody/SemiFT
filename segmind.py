import argparse

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset.semi_segmind import SemiDataset
from dataset.val import ValDataset
from supervised import validation_cpu
from util.classes import CLASSES
from util.segmind_utils import MemoryBank, contrastive_loss, generate_u_data, get_batch_mask_tensor
from util.ssl_method_utils import (
    build_criterions,
    build_ema_model,
    build_logger_and_runtime,
    build_model,
    build_optimizer,
    log_model_info,
    maybe_load_checkpoint,
    save_checkpoint,
    update_ema,
    update_lr,
    wrap_ddp,
)
from util.utils import AverageMeter


def get_parser():
    parser = argparse.ArgumentParser(description="SegMind with DPT/DINO backbone")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def build_dataloaders(args, cfg):
    trainset_u = SemiDataset(
        cfg["dataset"], cfg["data_root"], "train_u", cfg["crop_size"], args.unlabeled_id_path, ignore_index=cfg["ignore_index"]
    )
    trainset_l = SemiDataset(
        cfg["dataset"], cfg["data_root"], "train_l", cfg["crop_size"], args.labeled_id_path, nsample=len(trainset_u.ids), ignore_index=cfg["ignore_index"]
    )
    valset = ValDataset(cfg["dataset"], cfg["data_root"], "val", ignore_value=cfg["ignore_index"])
    workers = cfg.get("workers", 4)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg["batch_size"], sampler=torch.utils.data.distributed.DistributedSampler(trainset_l), pin_memory=True, num_workers=workers, drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg["batch_size"], sampler=torch.utils.data.distributed.DistributedSampler(trainset_u), pin_memory=True, num_workers=workers, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, sampler=torch.utils.data.distributed.DistributedSampler(valset), pin_memory=True, num_workers=1, drop_last=False)
    return trainloader_l, trainloader_u, valloader


def get_cfg(cfg):
    seg_cfg = {
        "lambda_l": 1.0,
        "lambda_e": 1.0,
        "lambda_r": 1.0,
        "lambda_rsc": 1.0,
        "lambda_c": 1.0,
        "pseudo_threshold": 0.7,
        "query_threshold": 0.97,
        "temperature": 0.5,
        "bank_size": 10000,
        "num_query": 256,
        "num_negative": 512,
        "epoch_pre": 50,
        "mask_rate": 0.25,
        "mask_gap": 4,
        "proj_dim": 256,
    }
    seg_cfg.update(cfg.get("segmind", {}))
    seg_cfg["class_num"] = cfg["nclass"]
    return seg_cfg


def main(args, cfg):
    logger, rank, world_size, writer = build_logger_and_runtime(args, cfg)
    seg_cfg = get_cfg(cfg)
    model, load_result = build_model(cfg, method="segmind")
    optimizer = build_optimizer(cfg, model)
    log_model_info(logger, rank, model, load_result)
    model, local_rank = wrap_ddp(model)
    model_ema = build_ema_model(model)
    criterion_l, criterion_u = build_criterions(cfg, local_rank)
    trainloader_l, trainloader_u, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader_u) * cfg["epochs"]
    memory_bank = MemoryBank(cfg["nclass"], seg_cfg["bank_size"], seg_cfg["proj_dim"])

    state = maybe_load_checkpoint(args, model, optimizer, model_ema=model_ema)
    previous_best = state["previous_best"]
    previous_best_ema = state["previous_best_ema"]
    best_epoch = state["best_epoch"]
    best_epoch_ema = state["best_epoch_ema"]
    start_epoch = state["epoch"]

    for epoch in range(start_epoch + 1, cfg["epochs"]):
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        model.train()
        loss_meters = {name: AverageMeter() for name in ["loss", "l", "e", "r", "rsc", "c"]}

        for i, ((rs_l_w, rs_l_s, lab_l), (rs_u_w, rs_u_s, _)) in enumerate(zip(trainloader_l, trainloader_u)):
            rs_l_w = rs_l_w.cuda()
            rs_l_s = rs_l_s.cuda()
            lab_l = lab_l.cuda()
            rs_u_w = rs_u_w.cuda()
            rs_u_s = rs_u_s.cuda()
            hw = lab_l.shape[-2:]
            n_l = lab_l.shape[0]

            with torch.no_grad():
                teacher_in = torch.cat((rs_l_w, rs_u_w), dim=0)
                t_logits_all = model_ema(teacher_in).detach()
                t_prob_all = torch.softmax(t_logits_all, dim=1)
                pseudo_logit, pseudo_label = torch.max(t_prob_all[n_l:], dim=1)
                t_entropy_all = torch.sum(-t_prob_all * torch.log(t_prob_all + 1e-8), dim=1)
                rs_u_w_mix, rs_u_s_mix, pseudo_label_mix, _, entropy_mix = generate_u_data(
                    rs_u_w, rs_u_s, pseudo_label, pseudo_logit, t_entropy_all[n_l:]
                )
                lab_all = torch.cat((lab_l, pseudo_label_mix), dim=0)
                rs_all_s = torch.cat((rs_l_s, rs_u_s_mix), dim=0)
                rs_all_w = torch.cat((rs_l_w, rs_u_w_mix), dim=0)
                t_entropy_target = torch.cat((t_entropy_all[:n_l], entropy_mix), dim=0)

            s_pred_all, s_feat_all, _ = model.module(rs_all_s, return_aux=True)
            loss_l = criterion_l(s_pred_all, lab_all)

            s_prob_all = torch.softmax(s_pred_all, dim=1)
            s_entropy_all = torch.sum(-s_prob_all * torch.log(s_prob_all + 1e-8), dim=1)
            loss_e = F.mse_loss(s_entropy_all, t_entropy_target)

            loss_r = torch.tensor(0.0, device=rs_all_w.device)
            loss_rsc = torch.tensor(0.0, device=rs_all_w.device)
            if epoch <= seg_cfg["epoch_pre"] and (seg_cfg["lambda_r"] != 0 or seg_cfg["lambda_rsc"] != 0):
                mask_tensor = get_batch_mask_tensor(
                    rs_all_w.shape,
                    mask_gap=seg_cfg["mask_gap"],
                    mask_rate=seg_cfg["mask_rate"],
                    device=rs_all_w.device,
                )
                r_logits, _, r_recon = model.module(rs_all_w * mask_tensor, mode="r", mask=mask_tensor)
                masked_pixels = ~mask_tensor.bool().expand_as(r_recon)
                if masked_pixels.any():
                    loss_r = F.mse_loss(r_recon[masked_pixels], rs_all_w[masked_pixels])
                masked_seg_pixels = (~mask_tensor.bool()).squeeze(1)
                if masked_seg_pixels.any():
                    loss_rsc = F.cross_entropy(r_logits.permute(0, 2, 3, 1)[masked_seg_pixels], lab_all[masked_seg_pixels], ignore_index=cfg["ignore_index"])

            feat_hw = s_feat_all.shape[-2:]
            lab_small = F.interpolate(lab_all.float().unsqueeze(1), size=feat_hw, mode="nearest").long().squeeze(1)
            prob_small = F.interpolate(s_prob_all, size=feat_hw, mode="bilinear", align_corners=True)
            loss_c = contrastive_loss(s_feat_all, lab_small, prob_small, seg_cfg, memory_bank)

            loss = (
                seg_cfg["lambda_l"] * loss_l
                + seg_cfg["lambda_e"] * loss_e
                + seg_cfg["lambda_r"] * loss_r
                + seg_cfg["lambda_rsc"] * loss_rsc
                + seg_cfg["lambda_c"] * loss_c
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iters = epoch * len(trainloader_u) + i
            lr = update_lr(optimizer, cfg, iters, total_iters)
            update_ema(model, model_ema, iters, max_decay=cfg.get("segmind", {}).get("ema_decay", 0.99))

            loss_meters["loss"].update(loss.item())
            loss_meters["l"].update(loss_l.item())
            loss_meters["e"].update(loss_e.item())
            loss_meters["r"].update(loss_r.item())
            loss_meters["rsc"].update(loss_rsc.item())
            loss_meters["c"].update(loss_c.item())

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_l", loss_l.item(), iters)
                writer.add_scalar("train/loss_e", loss_e.item(), iters)
                writer.add_scalar("train/loss_r", loss_r.item(), iters)
                writer.add_scalar("train/loss_rsc", loss_rsc.item(), iters)
                writer.add_scalar("train/loss_c", loss_c.item(), iters)
                if i % max(1, len(trainloader_u) // 8) == 0:
                    logger.info(
                        "Iters: %d, LR: %.7f, Loss: %.3f, L: %.3f, E: %.3f, R: %.3f, RSC: %.3f, C: %.3f",
                        i,
                        lr,
                        loss_meters["loss"].avg,
                        loss_meters["l"].avg,
                        loss_meters["e"].avg,
                        loss_meters["r"].avg,
                        loss_meters["rsc"].avg,
                        loss_meters["c"].avg,
                    )

        mIoU, iou_class = validation_cpu(cfg, model, valloader)
        mIoU_ema, iou_class_ema = validation_cpu(cfg, model_ema, valloader)
        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [%d %s] IoU: %.2f, EMA: %.2f",
                    cls_idx,
                    CLASSES[cfg["dataset"]][cls_idx],
                    iou,
                    iou_class_ema[cls_idx],
                )
            logger.info("***** Evaluation ***** >>>> MeanIoU: %.2f, EMA: %.2f\n", mIoU, mIoU_ema)
            writer.add_scalar("eval/mIoU", mIoU, epoch)
            writer.add_scalar("eval/mIoU_ema", mIoU_ema, epoch)

        is_best = mIoU >= previous_best
        previous_best = max(previous_best, mIoU)
        previous_best_ema = max(previous_best_ema, mIoU_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        save_checkpoint(
            args,
            rank,
            model,
            optimizer,
            epoch,
            previous_best,
            best_epoch,
            model_ema=model_ema,
            previous_best_ema=previous_best_ema,
            best_epoch_ema=best_epoch_ema,
            is_best=is_best,
        )


if __name__ == "__main__":
    args = get_parser()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    main(args, cfg)
