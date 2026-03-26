import argparse
from copy import deepcopy
from datetime import datetime
import logging
import os
import pprint
import random

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from dataset.transform import blur, crop, hflip, normalize, resize
from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.upernet import UperNet
from util.classes import CLASSES
from util.focal import FocalLoss
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import AverageMeter, count_params, init_log
from util.dist_helper import setup_distributed
from util.ssl_method_utils import (
    get_local_rank,
    load_checkpoint_on_cpu,
    save_checkpoint_to_disk,
    log_cuda_memory,
    get_model_kwargs,
    get_backbone_info,
    load_backbone_checkpoint,
)
from util.validation import validation_cpu as shared_validation_cpu
from util.viz import Visualizer


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None


class SegMindLabeledDataset(Dataset):
    def __init__(
        self, name, root, mode, size=None, id_path=None, nsample=None, ignore_index=255
    ):
        assert mode == "train_l"
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_index = ignore_index
        with open(id_path, "r") as f:
            self.ids = f.read().splitlines()
        if nsample is not None and nsample > len(self.ids):
            self.ids *= int(np.ceil(float(nsample) / float(len(self.ids))))
            self.ids = self.ids[:nsample]
        self.strong_aug = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

    def __getitem__(self, item):
        sample_id = random.choice(self.ids)
        img = Image.open(os.path.join(self.root, sample_id.split(" ")[0])).convert("RGB")
        mask = Image.fromarray(
            np.array(Image.open(os.path.join(self.root, sample_id.split(" ")[1])))
        )

        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size, self.ignore_index)
        img, mask = hflip(img, mask, p=0.5)

        img_w = normalize(img)
        img_s = self.strong_aug(img.copy())
        img_s = blur(img_s, p=0.5)
        img_s = normalize(img_s)
        mask = torch.from_numpy(np.array(mask)).long()
        return img_w, img_s, mask

    def __len__(self):
        return len(self.ids) * 50


class SegMindUnlabeledDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, ignore_index=255):
        assert mode == "train_u"
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_index = ignore_index
        with open(id_path, "r") as f:
            self.ids = f.read().splitlines()
        self.strong_aug = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

    def __getitem__(self, item):
        sample_id = random.choice(self.ids)
        img = Image.open(os.path.join(self.root, sample_id.split(" ")[0])).convert("RGB")
        mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))

        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size, 254)
        img, mask = hflip(img, mask, p=0.5)

        img_w = normalize(img)
        img_s = self.strong_aug(img.copy())
        img_s = blur(img_s, p=0.5)
        img_s = normalize(img_s)

        mask = torch.from_numpy(np.array(mask)).long()
        valid_mask = torch.zeros_like(mask)
        valid_mask[mask == 254] = 255
        return img_w, img_s, valid_mask

    def __len__(self):
        return len(self.ids) * 50


class SegMindAuxHeads(nn.Module):
    def __init__(self, feat_channels, nclass):
        super().__init__()
        hidden = max(feat_channels // 2, 32)
        self.recon_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )
        self.recon_seg_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(feat_channels, nclass, kernel_size=1),
        )

    def forward(self, feats, image_size):
        recon = self.recon_head(feats)
        recon_seg = self.recon_seg_head(feats)
        recon = F.interpolate(recon, size=image_size, mode="bilinear", align_corners=False)
        recon_seg = F.interpolate(
            recon_seg, size=image_size, mode="bilinear", align_corners=False
        )
        return recon, recon_seg


@torch.no_grad()
def validation_cpu(cfg, model, valid_loader):
    return shared_validation_cpu(cfg, model, valid_loader)


def get_parser():
    parser = argparse.ArgumentParser(
        description="SegMind-style semi-supervised segmentation adapted to SemiFT"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def get_segmind_cfg(cfg):
    defaults = {
        "pseudo_threshold": 0.7,
        "ema_decay": 0.99,
        "lambda_x": 1.0,
        "lambda_u": 1.0,
        "lambda_e": 1.0,
        "lambda_r": 1.0,
        "lambda_rsc": 1.0,
        "lambda_c": 1.0,
        "pretrain_epochs": 50,
        "query_threshold": 0.97,
        "temperature": 0.5,
        "bank_size": 10000,
        "num_query": 256,
        "num_negative": 512,
        "mask_ratio": 0.25,
        "mask_patch": 16,
    }
    merged = dict(defaults)
    merged.update(cfg.get("segmind", {}))
    return merged


def get_base_model(model):
    return model.module if hasattr(model, "module") else model


def infer_feat_channels(base_model):
    if isinstance(base_model, DPT):
        conv = base_model.head.scratch.output_conv[0]
        return conv.in_channels
    if isinstance(base_model, UperNet):
        return base_model.decoder.classifier.in_channels
    raise TypeError(f"Unsupported model type: {type(base_model)!r}")


def forward_logits_and_feats(model, x):
    base_model = get_base_model(model)
    if isinstance(base_model, DPT):
        features, patch_h, patch_w = base_model._extract_features(x)
        logits, feats = base_model.head(features, patch_h, patch_w, return_feats=True)
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=True
        )
        return logits, feats
    if isinstance(base_model, UperNet):
        feat_maps = base_model._extract_feature_maps(x)
        pyramid_feats = base_model.neck(feat_maps)
        logits, feats = base_model.decoder(pyramid_feats, return_feats=True)
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits, feats
    raise TypeError(f"Unsupported model type: {type(base_model)!r}")


def entropy_map(prob):
    return -(prob.clamp(min=1e-8) * prob.clamp(min=1e-8).log()).sum(dim=1)


def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)
    if labels.numel() == 0:
        return torch.ones_like(pseudo_labels, dtype=torch.float32)
    keep = max(1, labels.numel() // 2)
    labels_select = labels[torch.randperm(labels.numel(), device=labels.device)[:keep]]
    return (pseudo_labels.unsqueeze(-1) == labels_select).any(-1).float()


def classmix_batch(img_w, img_s, pseudo_label, pseudo_conf, pseudo_entropy, valid_mask):
    batch_size = img_w.shape[0]
    device = img_w.device
    mixed_img_w, mixed_img_s = [], []
    mixed_lab, mixed_conf, mixed_entropy, mixed_valid = [], [], [], []
    for i in range(batch_size):
        mix_mask = generate_class_mask(pseudo_label[i]).to(device)
        src = (i + 1) % batch_size
        inv_mask = 1.0 - mix_mask
        mixed_img_w.append(
            (img_w[i] * mix_mask.unsqueeze(0) + img_w[src] * inv_mask.unsqueeze(0)).unsqueeze(0)
        )
        mixed_img_s.append(
            (img_s[i] * mix_mask.unsqueeze(0) + img_s[src] * inv_mask.unsqueeze(0)).unsqueeze(0)
        )
        mixed_lab.append(
            (pseudo_label[i] * mix_mask + pseudo_label[src] * inv_mask).unsqueeze(0)
        )
        mixed_conf.append(
            (pseudo_conf[i] * mix_mask + pseudo_conf[src] * inv_mask).unsqueeze(0)
        )
        mixed_entropy.append(
            (pseudo_entropy[i] * mix_mask + pseudo_entropy[src] * inv_mask).unsqueeze(0)
        )
        mixed_valid.append(
            torch.where(mix_mask.bool(), valid_mask[i], valid_mask[src]).unsqueeze(0)
        )
    return (
        torch.cat(mixed_img_w, dim=0),
        torch.cat(mixed_img_s, dim=0),
        torch.cat(mixed_lab, dim=0).long(),
        torch.cat(mixed_conf, dim=0),
        torch.cat(mixed_entropy, dim=0),
        torch.cat(mixed_valid, dim=0),
    )


def create_block_mask(batch_size, height, width, mask_patch=16, mask_ratio=0.25, device=None):
    gh = int(np.ceil(float(height) / float(mask_patch)))
    gw = int(np.ceil(float(width) / float(mask_patch)))
    mask_small = (torch.rand(batch_size, 1, gh, gw, device=device) < mask_ratio).float()
    mask = F.interpolate(mask_small, size=(height, width), mode="nearest")
    return mask


def reconstruction_loss(recon_img, target_img, mask):
    masked = mask.expand_as(recon_img)
    denom = masked.sum().clamp(min=1.0)
    return (((recon_img - target_img) ** 2) * masked).sum() / denom


def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    if keys.numel() == 0:
        return queue, queue_ptr
    ptr = int(queue_ptr.item()) if torch.is_tensor(queue_ptr) else int(queue_ptr)
    queue = torch.cat((queue, keys.detach()), dim=0)
    if queue.shape[0] >= queue_size:
        queue = queue[-queue_size:]
        ptr = queue_size
    else:
        ptr = (ptr + keys.shape[0]) % queue_size
    return queue, torch.tensor(ptr, device=keys.device, dtype=torch.long)


def get_negative_feat(samp_num, memory_bank_list, num_query, num_negative, feat_dim, device):
    negative_feat_all = torch.zeros((num_query, num_negative, feat_dim), device=device)
    for i in range(samp_num.shape[0]):
        negative_feat_i_list = []
        for j in range(samp_num.shape[1]):
            count = int(samp_num[i, j].item())
            if count <= 0 or memory_bank_list[j].shape[0] == 0:
                continue
            rand_idx = torch.randint(0, memory_bank_list[j].shape[0], (count,), device=device)
            negative_feat_i_list.append(memory_bank_list[j][rand_idx])
        if not negative_feat_i_list:
            continue
        negative_feat_i = torch.cat(negative_feat_i_list, dim=0)
        negative_feat_all[i, : negative_feat_i.shape[0]] = negative_feat_i[:num_negative]
    return negative_feat_all


def contrastive_loss(
    feat,
    lab,
    prob,
    memory_bank_list,
    queue_ptr_list,
    cfg_seg,
):
    device = feat.device
    feat = feat.permute(0, 2, 3, 1)
    feat_dim = feat.shape[-1]
    class_num = prob.shape[1]
    loss_c = feat.new_tensor(0.0)
    valid_classes = []
    feat_mean_batch_list = []
    feat_hard_batch_list = []
    feat_mean_set = feat.new_zeros((class_num, feat_dim))

    for class_i in range(class_num):
        lab_i = lab == class_i
        if lab_i.sum() == 0:
            continue
        valid_classes.append(class_i)
        prob_i = prob[:, class_i, :, :]
        feat_i_hard_mask = (prob_i < cfg_seg["query_threshold"]) & lab_i

        updated_queue, updated_ptr = dequeue_and_enqueue(
            feat[lab_i],
            memory_bank_list[class_i],
            queue_ptr_list[class_i],
            cfg_seg["bank_size"],
        )
        memory_bank_list[class_i] = updated_queue
        queue_ptr_list[class_i] = updated_ptr

        feat_mean_batch_list.append(torch.mean(feat[lab_i], dim=0, keepdim=True))
        feat_hard_batch_list.append(feat[feat_i_hard_mask])
        if memory_bank_list[class_i].shape[0] > 0:
            feat_mean_set[class_i] = torch.mean(memory_bank_list[class_i], dim=0)

    if not valid_classes:
        return loss_c, memory_bank_list, queue_ptr_list

    for idx, class_i in enumerate(valid_classes):
        if feat_hard_batch_list[idx].shape[0] == 0:
            continue
        hard_idx = torch.randint(
            feat_hard_batch_list[idx].shape[0],
            (cfg_seg["num_query"],),
            device=device,
        )
        query_feat = feat_hard_batch_list[idx][hard_idx]

        with torch.no_grad():
            sim = F.cosine_similarity(feat_mean_batch_list[idx], feat_mean_set, dim=1)
            if sim.numel() <= 1:
                continue
            sim[class_i] = -1e4
            neg_prob = torch.softmax(sim, dim=0)
            sample_class = torch.distributions.Categorical(probs=neg_prob).sample(
                (cfg_seg["num_query"], cfg_seg["num_negative"])
            )
            sample_class_num = torch.stack(
                [(sample_class == c).sum(1) for c in range(class_num)], dim=1
            )
            negative_feat = get_negative_feat(
                sample_class_num,
                memory_bank_list,
                cfg_seg["num_query"],
                cfg_seg["num_negative"],
                feat_dim,
                device,
            )
            positive_feat = feat_mean_batch_list[idx].unsqueeze(0).repeat(
                cfg_seg["num_query"], 1, 1
            )
            all_feat = torch.cat((positive_feat, negative_feat), dim=1)

        seg_logits = F.cosine_similarity(query_feat.unsqueeze(1), all_feat, dim=2)
        loss_c = loss_c + F.cross_entropy(
            seg_logits / cfg_seg["temperature"],
            torch.zeros(cfg_seg["num_query"], dtype=torch.long, device=device),
        )

    loss_c = loss_c / max(len(valid_classes), 1)
    return loss_c, memory_bank_list, queue_ptr_list


def build_model(cfg):
    model_kwargs = get_model_kwargs(cfg)
    _, backbone_version = get_backbone_info(cfg)
    if cfg["model"] == "dpt":
        model = DPT(**model_kwargs, backbone_version=backbone_version)
    elif cfg["model"] == "upernet":
        model = UperNet(**model_kwargs, backbone_version=backbone_version)
    else:
        raise ValueError(f"Unsupported model: {cfg['model']}")
    load_result = load_backbone_checkpoint(model, cfg)
    if cfg["lock_backbone"]:
        model.lock_backbone()
    return model, load_result


def build_optimizer(cfg, model, aux_heads):
    aux_params = [p for p in aux_heads.parameters() if p.requires_grad]
    return AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": cfg["lr"],
            },
            {
                "params": [
                    p for name, p in model.named_parameters() if "backbone" not in name
                ],
                "lr": cfg["lr"] * cfg.get("lr_multi", 1.0),
            },
            {
                "params": aux_params,
                "lr": cfg["lr"] * cfg.get("lr_multi", 1.0),
            },
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )


def build_criterion(cfg, local_rank):
    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "FocalLoss":
        criterion_l = FocalLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    else:
        raise NotImplementedError(cfg["criterion"]["name"])
    criterion_u = nn.CrossEntropyLoss(reduction="none", ignore_index=cfg["ignore_index"]).cuda(local_rank)
    criterion_e = nn.MSELoss().cuda(local_rank)
    return criterion_l, criterion_u, criterion_e


def main(args, cfg):
    logger = init_log("global", logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))
        os.makedirs(args.save_path, exist_ok=True)
        if SummaryWriter is None:
            logger.warning("tensorboard is not installed; scalar logging is disabled.")
            writer = NullWriter()
        else:
            writer = SummaryWriter(args.save_path)
    else:
        writer = NullWriter()

    cudnn.enabled = True
    cudnn.benchmark = True

    seg_cfg = get_segmind_cfg(cfg)

    model, load_result = build_model(cfg)
    feat_channels = infer_feat_channels(model)
    aux_heads = SegMindAuxHeads(feat_channels, cfg["nclass"])
    optimizer = build_optimizer(cfg, model, aux_heads)

    if rank == 0:
        logger.info("Total params: {:.1f}M".format(count_params(model)))
        logger.info("Encoder params: {:.1f}M".format(count_params(model.backbone)))
        logger.info("Decoder params: {:.1f}M".format(count_params(model.head)))
        logger.info("Aux head params: {:.1f}M".format(count_params(aux_heads)))
        logger.info(
            "backbone load_result missing_keys=%s unexpected_keys=%s",
            list(getattr(load_result, "missing_keys", [])),
            list(getattr(load_result, "unexpected_keys", [])),
        )

    local_rank = get_local_rank()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    aux_heads = torch.nn.SyncBatchNorm.convert_sync_batchnorm(aux_heads)
    model.cuda(local_rank)
    aux_heads.cuda(local_rank)
    log_cuda_memory(logger, rank, "after_model_to_cuda", local_rank=local_rank, save_path=args.save_path)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )
    aux_heads = torch.nn.parallel.DistributedDataParallel(
        aux_heads,
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

    criterion_l, criterion_u, criterion_e = build_criterion(cfg, local_rank)

    trainset_u = SegMindUnlabeledDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        ignore_index=cfg["ignore_index"],
    )
    trainset_l = SegMindLabeledDataset(
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

    workers = cfg.get("workers", 4)
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
        drop_last=True,
        sampler=trainsampler_l,
    )
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=workers,
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
    best_epoch, best_epoch_ema = -1, -1
    start_epoch = 0

    memory_bank_list = [
        torch.zeros((0, feat_channels), device=local_rank) for _ in range(cfg["nclass"])
    ]
    queue_ptr_list = [
        torch.zeros(1, device=local_rank, dtype=torch.long) for _ in range(cfg["nclass"])
    ]

    latest_path = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(latest_path):
        log_cuda_memory(logger, rank, "before_resume_load", save_path=args.save_path)
        checkpoint = load_checkpoint_on_cpu(latest_path)
        model.load_state_dict(checkpoint["model"])
        model_ema.load_state_dict(checkpoint["model_ema"])
        aux_heads.load_state_dict(checkpoint["aux_heads"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        log_cuda_memory(logger, rank, "after_resume_load", save_path=args.save_path)
        start_epoch = checkpoint["epoch"] + 1
        previous_best = checkpoint.get("previous_best", 0.0)
        previous_best_ema = checkpoint.get("previous_best_ema", 0.0)
        best_epoch = checkpoint.get("best_epoch", -1)
        best_epoch_ema = checkpoint.get("best_epoch_ema", -1)
        if "memory_bank_list" in checkpoint:
            memory_bank_list = [bank.cuda(local_rank) for bank in checkpoint["memory_bank_list"]]
        if "queue_ptr_list" in checkpoint:
            queue_ptr_list = [ptr.cuda(local_rank) for ptr in checkpoint["queue_ptr_list"]]
        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % (start_epoch - 1))

    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz = Visualizer(save_dir=f"./viz/{filename}", dataset=cfg["dataset"])

    for epoch in range(start_epoch, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, EMA: {:.2f} @epoch-{:}".format(
                    epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema
                )
            )

        trainsampler_l.set_epoch(epoch)
        trainsampler_u.set_epoch(epoch)
        model.train()
        aux_heads.train()

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_u = AverageMeter()
        total_loss_e = AverageMeter()
        total_loss_r = AverageMeter()
        total_loss_rsc = AverageMeter()
        total_loss_c = AverageMeter()
        total_mask_ratio = AverageMeter()

        for i, ((img_x_w, img_x_s, mask_x), (img_u_w, img_u_s, ignore_mask_u)) in enumerate(
            zip(trainloader_l, trainloader_u)
        ):
            img_x_w = img_x_w.cuda(local_rank)
            img_x_s = img_x_s.cuda(local_rank)
            mask_x = mask_x.cuda(local_rank)
            img_u_w = img_u_w.cuda(local_rank)
            img_u_s = img_u_s.cuda(local_rank)
            ignore_mask_u = ignore_mask_u.cuda(local_rank)
            valid_mask_u = ignore_mask_u != 255

            with torch.no_grad():
                teacher_logits_all, _ = forward_logits_and_feats(
                    model_ema, torch.cat((img_x_w, img_u_w), dim=0)
                )
                teacher_prob_all = teacher_logits_all.softmax(dim=1)
                teacher_entropy_all = entropy_map(teacher_prob_all)
                num_lb = img_x_w.shape[0]
                prob_u = teacher_prob_all[num_lb:]
                entropy_u = teacher_entropy_all[num_lb:]
                conf_u, pseudo_u = prob_u.max(dim=1)

                (
                    img_u_w_mix,
                    img_u_s_mix,
                    pseudo_u_mix,
                    conf_u_mix,
                    entropy_u_mix,
                    valid_mask_u_mix,
                ) = classmix_batch(
                    img_u_w, img_u_s, pseudo_u, conf_u, entropy_u, valid_mask_u
                )

                teacher_entropy_mix_all = torch.cat(
                    (teacher_entropy_all[:num_lb], entropy_u_mix), dim=0
                )

            logits_all, feats_all = forward_logits_and_feats(
                model, torch.cat((img_x_s, img_u_s_mix), dim=0)
            )
            prob_all = logits_all.softmax(dim=1)
            pred_x = logits_all[:num_lb]
            pred_u = logits_all[num_lb:]
            prob_u_student = prob_all[num_lb:]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_map = criterion_u(pred_u, pseudo_u_mix)
            unsup_mask = (conf_u_mix >= seg_cfg["pseudo_threshold"]) & valid_mask_u_mix
            loss_u = (loss_u_map * unsup_mask.float()).sum() / unsup_mask.sum().clamp(min=1.0)

            student_entropy_all = entropy_map(prob_all)
            loss_e = criterion_e(student_entropy_all, teacher_entropy_mix_all)

            loss_r = logits_all.new_tensor(0.0)
            loss_rsc = logits_all.new_tensor(0.0)
            if epoch < seg_cfg["pretrain_epochs"]:
                weak_all = torch.cat((img_x_w, img_u_w_mix), dim=0)
                mask_block = create_block_mask(
                    weak_all.shape[0],
                    weak_all.shape[-2],
                    weak_all.shape[-1],
                    mask_patch=seg_cfg["mask_patch"],
                    mask_ratio=seg_cfg["mask_ratio"],
                    device=weak_all.device,
                )
                masked_inputs = weak_all * (1.0 - mask_block)
                _, masked_feats = forward_logits_and_feats(model, masked_inputs)
                recon_img, recon_seg = aux_heads(masked_feats, weak_all.shape[-2:])
                loss_r = reconstruction_loss(recon_img, weak_all, mask_block)

                pseudo_all = torch.cat((mask_x, pseudo_u_mix), dim=0)
                recon_seg_loss_map = criterion_u(recon_seg, pseudo_all)
                valid_all = torch.cat(
                    (torch.ones_like(mask_x, dtype=torch.bool), valid_mask_u_mix), dim=0
                )
                conf_all = torch.cat(
                    (torch.ones_like(mask_x, dtype=conf_u_mix.dtype), conf_u_mix), dim=0
                )
                rsc_mask = valid_all & (conf_all >= seg_cfg["pseudo_threshold"])
                loss_rsc = (recon_seg_loss_map * rsc_mask.float()).sum() / rsc_mask.sum().clamp(min=1.0)

            feat_h, feat_w = feats_all.shape[-2:]
            pseudo_all_small = F.interpolate(
                torch.cat((mask_x, pseudo_u_mix), dim=0).float().unsqueeze(1),
                size=(feat_h, feat_w),
                mode="nearest",
            ).squeeze(1).long()
            prob_all_small = F.interpolate(
                prob_all,
                size=(feat_h, feat_w),
                mode="bilinear",
                align_corners=False,
            )
            loss_c, memory_bank_list, queue_ptr_list = contrastive_loss(
                feats_all,
                pseudo_all_small,
                prob_all_small,
                memory_bank_list,
                queue_ptr_list,
                seg_cfg,
            )

            loss = (
                seg_cfg["lambda_x"] * loss_x
                + seg_cfg["lambda_u"] * loss_u
                + seg_cfg["lambda_e"] * loss_e
                + seg_cfg["lambda_r"] * loss_r
                + seg_cfg["lambda_rsc"] * loss_rsc
                + seg_cfg["lambda_c"] * loss_c
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader_u) + i
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg.get("lr_multi", 1.0)
            optimizer.param_groups[2]["lr"] = lr * cfg.get("lr_multi", 1.0)

            ema_ratio = min(1 - 1 / (iters + 1), seg_cfg["ema_decay"])
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            if i < 10:
                viz.push(
                    {
                        "img_x": (img_x_s[0], Visualizer.TENSOR),
                        "mask_x": (mask_x[0], Visualizer.SEGMENTATION),
                        "pred_x": (pred_x.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                        "img_u_s": (img_u_s_mix[0], Visualizer.TENSOR),
                        "pseudo_u": (pseudo_u_mix[0], Visualizer.SEGMENTATION),
                        "pred_u": (pred_u.argmax(dim=1)[0], Visualizer.SEGMENTATION),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_u.update(loss_u.item())
            total_loss_e.update(loss_e.item())
            total_loss_r.update(loss_r.item())
            total_loss_rsc.update(loss_rsc.item())
            total_loss_c.update(loss_c.item())
            mask_ratio = unsup_mask.float().sum() / valid_mask_u_mix.float().sum().clamp(min=1.0)
            total_mask_ratio.update(mask_ratio.item())

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_u", loss_u.item(), iters)
                writer.add_scalar("train/loss_e", loss_e.item(), iters)
                writer.add_scalar("train/loss_r", loss_r.item(), iters)
                writer.add_scalar("train/loss_rsc", loss_rsc.item(), iters)
                writer.add_scalar("train/loss_c", loss_c.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio.item(), iters)

            if (i % max(1, len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, LR: {:.7f}, Total: {:.3f}, X: {:.3f}, U: {:.3f}, E: {:.3f}, R: {:.3f}, RSC: {:.3f}, C: {:.3f}, Mask: {:.3f}".format(
                        i,
                        lr,
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_u.avg,
                        total_loss_e.avg,
                        total_loss_r.avg,
                        total_loss_rsc.avg,
                        total_loss_c.avg,
                        total_mask_ratio.avg,
                    )
                )

        val_cfg = dict(cfg)
        val_cfg.setdefault(
            "eval_mode", "slide_window" if cfg["dataset"] == "cityscapes" else "original"
        )
        val_cfg.setdefault("ignore_index", cfg.get("ignore_index", 255))
        eval_mode = val_cfg["eval_mode"]
        mIoU, iou_class = validation_cpu(val_cfg, model, valloader)
        mIoU_ema, iou_class_ema = validation_cpu(val_cfg, model_ema, valloader)

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
                writer.add_scalar("eval/%s_IoU" % CLASSES[cfg["dataset"]][j], iou, epoch)
                writer.add_scalar(
                    "eval/%s_IoU_ema" % CLASSES[cfg["dataset"]][j],
                    iou_class_ema[j],
                    epoch,
                )

        is_best = mIoU >= previous_best
        previous_best = max(previous_best, mIoU)
        previous_best_ema = max(previous_best_ema, mIoU_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch

        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "aux_heads": aux_heads.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
                "previous_best_ema": previous_best_ema,
                "best_epoch": best_epoch,
                "best_epoch_ema": best_epoch_ema,
                "memory_bank_list": memory_bank_list,
                "queue_ptr_list": queue_ptr_list,
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
