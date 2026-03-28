import argparse
from copy import deepcopy
import logging
import os
import pprint
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
import yaml

from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.ssl_method_utils import get_local_rank, load_checkpoint_on_cpu, save_checkpoint_to_disk, log_cuda_memory, load_backbone_checkpoint

DEFAULT_PEFT_CFG: Dict[str, Any] = {
    "method": "semift",
    "target_modules": ["mlp"],
    "freeze_backbone": True,
    "modules_to_save": ["head"],
    "bias": "lora_only",
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "ssf_init_scale": 1.0,
    "ssf_init_shift_std": 0.02,
    "adapter_dim": 64,
    "adapter_dropout": 0.1,
    "adapter_scale": 0.1,
    "adapter_layernorm_option": "none",
    "fact_rank": 8,
    "fact_scale": 1.0,
    "fact_dropout": 0.1,
    "conv_lora_num_experts": 4,
    "conv_lora_topk": 1,
    "conv_lora_kernel_size": 3,
    "conv_lora_dropout": 0.1,
    "hydra_num_branches": 4,
    "hydra_router_hidden": 64,
    "hydra_router_dropout": 0.1,
    "moe_num_experts": 4,
    "moe_topk": 2,
    "moe_router_balance_mode": "deepseek_v3",
    "moe_router_bias_update_speed": 1e-3,
    "moe_router_bias_clip": 0.05,
    "moe_router_aux_loss_coef": 1e-2,
    "moe_router_z_loss_coef": 1e-3,
    "moe_router_jitter_noise": 1e-2,
    "moe_num_prefix_tokens": -1,
    "moe_use_shared_expert": True,
    "moe_conv_hidden_ratio": 2.0,
    "moe_conv_kernel_size": 3,
    "moe_conv_context_kernel_size": 5,
    "moe_conv_use_grn": True,
    "moe_conv_norm_type": "layernorm",
    "moe_expert_scales": [1, 2, 4, 8],
    "moe_conv_gate_temperature": 1.0,
    "moe_layerscale_init": 1e-5,
    "moe_expert_drop_path_rate": 0.0,
    "moe_branch_gate_init_bias": -2.0,
}

METHOD_DEFAULT_TARGETS: Dict[str, List[str]] = {
    "semift": ["mlp"],
    "semift_samoe": ["mlp"],
    "samoev4": ["mlp"],
    "samoev5": ["mlp"],
    "samoev6": ["mlp"],
    "samoev7": ["mlp"],
    "samoev8": ["mlp"],
    "semift_scalegate": ["mlp"],
    "lora": ["qkv", "proj", "fc1", "fc2"],
    "ssf": ["patch_embed", "norm1", "norm2", "qkv", "proj", "fc1", "fc2"],
    "bitfit": ["qkv", "proj", "fc1", "fc2", "norm1", "norm2", "head"],
    "adaptformer": ["mlp"],
    "fact_tt": ["qkv", "proj", "fc1", "fc2"],
    "fact_tk": ["qkv", "proj", "fc1", "fc2"],
    "conv_lora": ["qkv", "proj", "fc1", "fc2"],
    "hydralora": ["qkv", "proj", "fc1", "fc2"],
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="UniMatchV2 + configurable PEFT for Semi-Supervised Semantic Segmentation"
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


def _normalize_target_modules(
    value: Optional[Union[str, List[str], tuple]]
) -> Union[str, List[str]]:
    if value is None:
        return list(DEFAULT_PEFT_CFG["target_modules"])
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return list(DEFAULT_PEFT_CFG["target_modules"])
        if "," in stripped:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return stripped
    if isinstance(value, (list, tuple)):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        if not normalized:
            return list(DEFAULT_PEFT_CFG["target_modules"])
        if len(normalized) == 1 and any(ch in normalized[0] for ch in "^$.*+?[](){}|\\"):
            return normalized[0]
        return normalized
    raise TypeError(f"Unsupported target_modules type: {type(value)!r}")


def resolve_peft_cfg(cfg: Dict[str, Any], args) -> Dict[str, Any]:
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
        peft_cfg["target_modules"] = _normalize_target_modules(peft_cfg.get("target_modules"))
    else:
        peft_cfg["target_modules"] = list(METHOD_DEFAULT_TARGETS[peft_cfg["method"]])

    cfg["peft"] = peft_cfg
    return peft_cfg


def build_peft_config(peft_cfg: Dict[str, Any], cfg: Dict[str, Any]):
    from peft.tuners.semift import SemiFTConfig

    return SemiFTConfig(
        method=peft_cfg["method"],
        target_modules=peft_cfg["target_modules"],
        modules_to_save=peft_cfg.get("modules_to_save"),
        bias=peft_cfg.get("bias", DEFAULT_PEFT_CFG["bias"]),
        nclass=cfg["nclass"],
        r=peft_cfg.get("r", DEFAULT_PEFT_CFG["r"]),
        lora_alpha=peft_cfg.get("lora_alpha", DEFAULT_PEFT_CFG["lora_alpha"]),
        lora_dropout=peft_cfg.get("lora_dropout", DEFAULT_PEFT_CFG["lora_dropout"]),
        ssf_init_scale=peft_cfg.get("ssf_init_scale", DEFAULT_PEFT_CFG["ssf_init_scale"]),
        ssf_init_shift_std=peft_cfg.get(
            "ssf_init_shift_std", DEFAULT_PEFT_CFG["ssf_init_shift_std"]
        ),
        adapter_dim=peft_cfg.get("adapter_dim", DEFAULT_PEFT_CFG["adapter_dim"]),
        adapter_dropout=peft_cfg.get(
            "adapter_dropout", DEFAULT_PEFT_CFG["adapter_dropout"]
        ),
        adapter_scale=peft_cfg.get("adapter_scale", DEFAULT_PEFT_CFG["adapter_scale"]),
        adapter_layernorm_option=peft_cfg.get(
            "adapter_layernorm_option",
            DEFAULT_PEFT_CFG["adapter_layernorm_option"],
        ),
        fact_rank=peft_cfg.get("fact_rank", DEFAULT_PEFT_CFG["fact_rank"]),
        fact_scale=peft_cfg.get("fact_scale", DEFAULT_PEFT_CFG["fact_scale"]),
        fact_dropout=peft_cfg.get("fact_dropout", DEFAULT_PEFT_CFG["fact_dropout"]),
        conv_lora_num_experts=peft_cfg.get(
            "conv_lora_num_experts", DEFAULT_PEFT_CFG["conv_lora_num_experts"]
        ),
        conv_lora_topk=peft_cfg.get(
            "conv_lora_topk", DEFAULT_PEFT_CFG["conv_lora_topk"]
        ),
        conv_lora_kernel_size=peft_cfg.get(
            "conv_lora_kernel_size", DEFAULT_PEFT_CFG["conv_lora_kernel_size"]
        ),
        conv_lora_dropout=peft_cfg.get(
            "conv_lora_dropout", DEFAULT_PEFT_CFG["conv_lora_dropout"]
        ),
        hydra_num_branches=peft_cfg.get(
            "hydra_num_branches", DEFAULT_PEFT_CFG["hydra_num_branches"]
        ),
        hydra_router_hidden=peft_cfg.get(
            "hydra_router_hidden", DEFAULT_PEFT_CFG["hydra_router_hidden"]
        ),
        hydra_router_dropout=peft_cfg.get(
            "hydra_router_dropout", DEFAULT_PEFT_CFG["hydra_router_dropout"]
        ),
        moe_num_experts=peft_cfg.get("moe_num_experts", DEFAULT_PEFT_CFG["moe_num_experts"]),
        moe_topk=peft_cfg.get("moe_topk", DEFAULT_PEFT_CFG["moe_topk"]),
        moe_router_balance_mode=peft_cfg.get(
            "moe_router_balance_mode", DEFAULT_PEFT_CFG["moe_router_balance_mode"]
        ),
        moe_router_bias_update_speed=peft_cfg.get(
            "moe_router_bias_update_speed",
            DEFAULT_PEFT_CFG["moe_router_bias_update_speed"],
        ),
        moe_router_bias_clip=peft_cfg.get(
            "moe_router_bias_clip", DEFAULT_PEFT_CFG["moe_router_bias_clip"]
        ),
        moe_router_aux_loss_coef=peft_cfg.get(
            "moe_router_aux_loss_coef", DEFAULT_PEFT_CFG["moe_router_aux_loss_coef"]
        ),
        moe_router_z_loss_coef=peft_cfg.get(
            "moe_router_z_loss_coef", DEFAULT_PEFT_CFG["moe_router_z_loss_coef"]
        ),
        moe_router_jitter_noise=peft_cfg.get(
            "moe_router_jitter_noise", DEFAULT_PEFT_CFG["moe_router_jitter_noise"]
        ),
        moe_num_prefix_tokens=peft_cfg.get(
            "moe_num_prefix_tokens", DEFAULT_PEFT_CFG["moe_num_prefix_tokens"]
        ),
        moe_use_shared_expert=peft_cfg.get(
            "moe_use_shared_expert", DEFAULT_PEFT_CFG["moe_use_shared_expert"]
        ),
        moe_conv_hidden_ratio=peft_cfg.get(
            "moe_conv_hidden_ratio", DEFAULT_PEFT_CFG["moe_conv_hidden_ratio"]
        ),
        moe_conv_kernel_size=peft_cfg.get(
            "moe_conv_kernel_size", DEFAULT_PEFT_CFG["moe_conv_kernel_size"]
        ),
        moe_conv_context_kernel_size=peft_cfg.get(
            "moe_conv_context_kernel_size",
            DEFAULT_PEFT_CFG["moe_conv_context_kernel_size"],
        ),
        moe_conv_use_grn=peft_cfg.get(
            "moe_conv_use_grn", DEFAULT_PEFT_CFG["moe_conv_use_grn"]
        ),
        moe_conv_norm_type=peft_cfg.get(
            "moe_conv_norm_type", DEFAULT_PEFT_CFG["moe_conv_norm_type"]
        ),
        moe_expert_scales=peft_cfg.get(
            "moe_expert_scales", DEFAULT_PEFT_CFG["moe_expert_scales"]
        ),
        moe_conv_gate_temperature=peft_cfg.get(
            "moe_conv_gate_temperature", DEFAULT_PEFT_CFG["moe_conv_gate_temperature"]
        ),
        moe_layerscale_init=peft_cfg.get(
            "moe_layerscale_init", DEFAULT_PEFT_CFG["moe_layerscale_init"]
        ),
        moe_expert_drop_path_rate=peft_cfg.get(
            "moe_expert_drop_path_rate", DEFAULT_PEFT_CFG["moe_expert_drop_path_rate"]
        ),
        moe_branch_gate_init_bias=peft_cfg.get(
            "moe_branch_gate_init_bias", DEFAULT_PEFT_CFG["moe_branch_gate_init_bias"]
        ),
    )


def apply_peft(model, peft_cfg: Dict[str, Any], cfg: Dict[str, Any]):
    from peft.tuners.semift import AdaptModel

    return AdaptModel(build_peft_config(peft_cfg, cfg), model)


def show_trainable_parameters(model, logger):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    trainable_params_names = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_params_names.append(name)

    percentage = 100 * trainable_params / total_params if total_params > 0 else 0

    logger.info("--- 模型可训练参数 ---")
    logger.info("--- 可训练模块/参数列表 ---")
    for name in trainable_params_names:
        logger.info(f" - {name}")
    logger.info("\n--- 统计信息 ---")
    logger.info(f" - 总参数数量: {total_params:,}")
    logger.info(f" - 可训练参数数量: {trainable_params:,}")
    logger.info(f" - 可训练参数占比: {percentage:.2f}%")


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
    from supervised import validation_cpu
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
            "Running UniMatchV2 + PEFT with method=%s, target_modules=%s, freeze_backbone=%s",
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
        if hasattr(model, "head"):
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

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
        ) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = (
                img_u_w.cuda(),
                img_u_s1.cuda(),
                img_u_s2.cuda(),
            )
            ignore_mask, cutmix_box1, cutmix_box2 = (
                ignore_mask.cuda(),
                cutmix_box1.cuda(),
                cutmix_box2.cuda(),
            )

            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = (
                img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            )
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = (
                img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            )

            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(
                torch.cat((img_u_s1, img_u_s2)), comp_drop=True
            ).chunk(2)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = (
                mask_u_w.clone(),
                conf_u_w.clone(),
                ignore_mask.clone(),
            )
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = (
                mask_u_w.clone(),
                conf_u_w.clone(),
                ignore_mask.clone(),
            )

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * (
                (conf_u_w_cutmixed1 >= cfg["conf_thresh"])
                & (ignore_mask_cutmixed1 != 255)
            )
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * (
                (conf_u_w_cutmixed2 >= cfg["conf_thresh"])
                & (ignore_mask_cutmixed2 != 255)
            )
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
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
                        "img_u_s1": (img_u_s1[0], Visualizer.TENSOR),
                        "img_u_s2": (img_u_s2[0], Visualizer.TENSOR),
                        "mask_u_w_cutmixed1": (
                            mask_u_w_cutmixed1[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "mask_u_w_cutmixed2": (
                            mask_u_w_cutmixed2[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_u_s1": (
                            pred_u_s1.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                        "pred_u_s2": (
                            pred_u_s2.argmax(dim=1)[0],
                            Visualizer.SEGMENTATION,
                        ),
                    }
                )
                viz.render(f"epoch_{epoch}_iter_{i}")
                viz.reset()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = (
                (conf_u_w >= cfg["conf_thresh"]) & (ignore_mask != 255)
            ).sum().item() / (ignore_mask != 255).sum()
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
