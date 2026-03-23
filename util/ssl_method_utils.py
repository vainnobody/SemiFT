import logging
import os
import pprint
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.semi_rs import SemiDataset as RemoteSemiDataset
from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.dpt_unimatch import DPT_UniMatch
from model.semseg.dpt_segmind import DPT_SegMind
from model.semseg.upernet import UperNet
from util.dist_helper import setup_distributed
from util.focal import FocalLoss
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log


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
    "resnet101": {
        "encoder_size": "resnet101",
        "features": 128,
        "out_channels": [256, 512, 1024, 2048],
    },
}


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None


def normalize_backbone_name(name):
    value = str(name).strip()
    lowered = value.lower().replace("-", "").replace("_", "")
    if lowered in {"rn101", "resnet101"}:
        return "resnet101"
    return value


def parse_backbone_spec(backbone_name):
    canonical = normalize_backbone_name(backbone_name)
    lowered = canonical.lower()
    if lowered == "resnet101":
        return {
            "canonical_name": "resnet101",
            "family": "resnet",
            "version": "resnet",
            "size": "resnet101",
        }

    parts = canonical.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Expected dinov2_*/dinov3_* or resnet101."
        )
    return {
        "canonical_name": canonical,
        "family": parts[0],
        "version": parts[0],
        "size": parts[-1],
    }


def get_model_kwargs(cfg):
    info = parse_backbone_spec(cfg["backbone"])
    model_cfg = MODEL_CONFIGS.get(info["size"])
    if model_cfg is None:
        raise ValueError(
            f"Unsupported backbone size '{info['size']}' for backbone '{cfg['backbone']}'."
        )
    return {**model_cfg, "nclass": cfg["nclass"]}


def get_backbone_checkpoint_path(cfg):
    info = parse_backbone_spec(cfg["backbone"])
    if info["family"] == "resnet":
        ckpt = cfg.get("backbone_ckpt")
        if ckpt:
            return ckpt
        default_ckpt = Path("./pretrained") / "resnet101.pth"
        if default_ckpt.exists():
            return str(default_ckpt)
        raise ValueError(
            "RN-101 backbone checkpoint not found. Set 'backbone_ckpt' in config "
            "or place the weights at './pretrained/resnet101.pth'."
        )
    return str(Path("./pretrained") / f"{normalize_backbone_name(cfg['backbone'])}.pth")


def _unwrap_checkpoint_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    for key in ("state_dict", "model", "module", "backbone"):
        value = state_dict.get(key)
        if isinstance(value, dict):
            state_dict = value
            break

    if not isinstance(state_dict, dict):
        return state_dict

    if any(k.startswith("backbone.") for k in state_dict.keys()):
        return {
            k[len("backbone.") :]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
    if any(k.startswith("module.backbone.") for k in state_dict.keys()):
        return {
            k[len("module.backbone.") :]: v
            for k, v in state_dict.items()
            if k.startswith("module.backbone.")
        }
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_backbone_checkpoint(model, cfg):
    ckpt_path = get_backbone_checkpoint_path(cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_checkpoint_state_dict(state_dict)
    return model.backbone.load_state_dict(state_dict, strict=False)


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))


def load_checkpoint_on_cpu(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: checkpoint_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [checkpoint_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_to_cpu(v) for v in value)
    return value


def save_checkpoint_to_disk(checkpoint, latest_path, best_path=None, is_best=False):
    cpu_checkpoint = checkpoint_to_cpu(checkpoint)
    latest_path = Path(latest_path)
    torch.save(cpu_checkpoint, latest_path)
    if is_best and best_path is not None:
        torch.save(cpu_checkpoint, Path(best_path))


def log_cuda_memory(logger, rank, stage, local_rank=None, save_path=None):
    if os.environ.get("SEMIFT_DDP_DEBUG_INIT", "0") != "1":
        return

    if local_rank is None:
        local_rank = get_local_rank()

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        allocated_gb = torch.cuda.memory_allocated(local_rank) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(local_rank) / 1024**3
        message = (
            f"[ddp-init] stage={stage} rank={rank} local_rank={local_rank} "
            f"current_device={current_device} allocated_gb={allocated_gb:.2f} "
            f"reserved_gb={reserved_gb:.2f}"
        )
    else:
        message = f"[ddp-init] stage={stage} rank={rank} local_rank={local_rank} cuda_unavailable"

    if save_path is not None:
        latest_path = Path(save_path) / "latest.pth"
        message += f" save_path={save_path} latest_exists={latest_path.exists()}"

    logger.info(message)


def build_logger_and_runtime(args, cfg):
    logger = init_log("global", logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)
    log_cuda_memory(logger, rank, "after_setup_distributed", save_path=args.save_path)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    else:
        writer = NullWriter()

    cudnn.enabled = True
    cudnn.benchmark = True
    return logger, rank, world_size, writer


def get_backbone_info(cfg):
    info = parse_backbone_spec(cfg["backbone"])
    return info["size"], info["version"]


def build_model(cfg, method="fixmatch"):
    _, backbone_version = get_backbone_info(cfg)
    kwargs = get_model_kwargs(cfg)

    if cfg["model"] == "upernet":
        model = UperNet(**kwargs, backbone_version=backbone_version)
    elif method == "unimatch":
        model = DPT_UniMatch(**kwargs, backbone_version=backbone_version)
    elif method == "segmind":
        segmind_cfg = cfg.get("segmind", {})
        model = DPT_SegMind(
            **kwargs,
            backbone_version=backbone_version,
            proj_dim=segmind_cfg.get("proj_dim", 256),
        )
    else:
        model = DPT(**kwargs, backbone_version=backbone_version)

    load_result = load_backbone_checkpoint(model, cfg)
    if cfg.get("lock_backbone"):
        model.lock_backbone()
    return model, load_result


def wrap_ddp(model, logger=None, rank=None, save_path=None):
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    if logger is not None and rank is not None:
        log_cuda_memory(logger, rank, "after_model_to_cuda", local_rank=local_rank, save_path=save_path)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )
    if logger is not None and rank is not None:
        log_cuda_memory(logger, rank, "after_ddp_wrap", local_rank=local_rank, save_path=save_path)
    return model, local_rank


def build_ema_model(model):
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    return model_ema


def build_optimizer(cfg, model):
    return AdamW(
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
                "lr": cfg["lr"] * cfg.get("lr_multi", 1.0),
            },
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )


def log_model_info(logger, rank, model, load_result=None):
    if rank != 0:
        return
    logger.info("Total params: {:.1f}M".format(count_params(model)))
    logger.info("Encoder params: {:.1f}M".format(count_params(model.backbone)))
    logger.info("Decoder params: {:.1f}M\n".format(count_params(model.head)))
    if load_result is not None:
        logger.info(
            "backbone load_result missing_keys=%s unexpected_keys=%s",
            list(getattr(load_result, "missing_keys", [])),
            list(getattr(load_result, "unexpected_keys", [])),
        )


def build_criterions(cfg, local_rank):
    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "FocalLoss":
        criterion_l = FocalLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    else:
        raise NotImplementedError(cfg["criterion"]["name"])
    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)
    return criterion_l, criterion_u


def build_standard_dataloaders(args, cfg, unlabeled_dataset=None, labeled_dataset=None):
    unlabeled_dataset = unlabeled_dataset or RemoteSemiDataset
    labeled_dataset = labeled_dataset or RemoteSemiDataset

    trainset_u = unlabeled_dataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        ignore_index=cfg["ignore_index"],
    )
    trainset_l = labeled_dataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(trainset_u.ids),
        ignore_index=cfg["ignore_index"],
    )
    valset = ValDataset(cfg["dataset"], cfg["data_root"], "val", ignore_value=cfg["ignore_index"])

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
    return trainloader_l, trainloader_u, valloader


def maybe_load_checkpoint(args, model, optimizer, model_ema=None, logger=None, rank=None):
    latest_path = os.path.join(args.save_path, "latest.pth")
    state = {
        "epoch": -1,
        "previous_best": 0.0,
        "previous_best_ema": 0.0,
        "best_epoch": 0,
        "best_epoch_ema": 0,
    }
    if not os.path.exists(latest_path):
        return state

    if logger is not None and rank is not None:
        log_cuda_memory(logger, rank, "before_resume_load", save_path=args.save_path)
    checkpoint = load_checkpoint_on_cpu(latest_path)
    model.load_state_dict(checkpoint["model"])
    if model_ema is not None and "model_ema" in checkpoint:
        model_ema.load_state_dict(checkpoint["model_ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if logger is not None and rank is not None:
        log_cuda_memory(logger, rank, "after_resume_load", save_path=args.save_path)
    for key in state:
        if key in checkpoint:
            state[key] = checkpoint[key]
    return state


def save_checkpoint(args, rank, model, optimizer, epoch, previous_best, best_epoch, model_ema=None, previous_best_ema=None, best_epoch_ema=None, extra=None, is_best=False):
    if rank != 0:
        return
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "previous_best": previous_best,
        "best_epoch": best_epoch,
    }
    if model_ema is not None:
        checkpoint["model_ema"] = model_ema.state_dict()
        checkpoint["previous_best_ema"] = previous_best_ema
        checkpoint["best_epoch_ema"] = best_epoch_ema
    if extra:
        checkpoint.update(extra)
    save_checkpoint_to_disk(
        checkpoint,
        os.path.join(args.save_path, "latest.pth"),
        os.path.join(args.save_path, "best.pth"),
        is_best=is_best,
    )


def update_ema(model, model_ema, iters, max_decay=0.996):
    ema_ratio = min(1 - 1 / (iters + 1), max_decay)
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
    for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
        buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))


def update_lr(optimizer, cfg, iters, total_iters):
    lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
    optimizer.param_groups[0]["lr"] = lr
    optimizer.param_groups[1]["lr"] = lr * cfg.get("lr_multi", 1.0)
    return lr
