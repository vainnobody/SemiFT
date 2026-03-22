import argparse
import csv
import json
import logging
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy import ndimage

from dataset.val import ValDataset
from model.semseg.dpt import DPT
from model.semseg.upernet import UperNet
from util.utils import color_map, intersectionAndUnion


LOGGER = logging.getLogger("cka")

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
}

DEFAULT_PEFT_CFG: Dict[str, object] = {
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
}

METHOD_DEFAULT_TARGETS: Dict[str, List[str]] = {
    "semift": ["mlp"],
    "semift_samoe": ["mlp"],
    "semift_scalegate": ["mlp"],
    "lora": ["qkv", "proj", "fc1", "fc2"],
    "ssf": ["patch_embed", "norm1", "norm2", "qkv", "proj", "fc1", "fc2"],
    "bitfit": ["qkv", "proj", "fc1", "fc2", "norm1", "norm2", "head"],
    "adaptformer": ["mlp"],
    "fact_tt": ["mlp"],
    "fact_tk": ["mlp"],
    "conv_lora": ["qkv", "proj", "fc1", "fc2"],
    "hydralora": ["qkv", "proj", "fc1", "fc2"],
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Single-checkpoint CKA + inference export script, plus summary mode "
            "for paper-style comparison across multiple finished runs."
        )
    )
    parser.add_argument("--summary-only", action="store_true")

    # Single-run mode
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--disable-peft", action="store_true")
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--sample-manifest", type=str, default=None)
    parser.add_argument(
        "--weights-source",
        type=str,
        default="auto",
        choices=["auto", "model", "model_ema"],
    )
    parser.add_argument(
        "--token-pool",
        type=str,
        default="mean_patch",
        choices=["mean_patch"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--save-inference-overlay",
        action="store_true",
        help="Save extra overlay images on top of the input image.",
    )

    # Summary mode
    parser.add_argument("--runs-root", type=str, default=None)
    parser.add_argument("--baseline-run", type=str, default=None)
    parser.add_argument("--summary-output-dir", type=str, default=None)
    parser.add_argument("--summary-topk", type=int, default=10)
    parser.add_argument(
        "--summary-metric",
        type=str,
        default="miou_gain",
        choices=["miou_gain", "error_reduction"],
    )
    parser.add_argument("--num-zoom-boxes", type=int, default=2)
    parser.add_argument("--zoom-pad", type=int, default=24)
    parser.add_argument("--overview-per-page", type=int, default=4)
    parser.add_argument("--min-component-area", type=int, default=16)
    return parser.parse_args()


def validate_args(args):
    if args.summary_only:
        if not args.runs_root or not args.baseline_run:
            raise ValueError("Summary mode requires --runs-root and --baseline-run.")
        return

    missing = [
        name
        for name in ("config", "checkpoint", "save_dir")
        if getattr(args, name) in (None, "")
    ]
    if missing:
        raise ValueError(f"Single-run mode missing required args: {missing}")
    if args.batch_size != 1:
        raise ValueError("CKA.py currently requires --batch-size 1 for variable-size segmentation samples.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)8s] %(message)s",
    )


def load_yaml(path: Union[str, Path]) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.Loader)


def normalize_target_modules(value: Optional[Union[str, Sequence[str]]]) -> Union[str, List[str]]:
    if value is None:
        return list(DEFAULT_PEFT_CFG["target_modules"])
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return list(DEFAULT_PEFT_CFG["target_modules"])
        if "," in stripped:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return stripped
    normalized = [str(item).strip() for item in value if str(item).strip()]
    if not normalized:
        return list(DEFAULT_PEFT_CFG["target_modules"])
    if len(normalized) == 1 and any(ch in normalized[0] for ch in "^$.*+?[](){}|\\"):
        return normalized[0]
    return normalized


def resolve_peft_cfg(cfg: Dict) -> Optional[Dict]:
    raw_yaml_peft = cfg.get("peft")
    if not raw_yaml_peft:
        return None
    peft_cfg = dict(DEFAULT_PEFT_CFG)
    peft_cfg.update(raw_yaml_peft)
    peft_cfg["method"] = str(peft_cfg["method"]).lower()
    if peft_cfg["method"] not in METHOD_DEFAULT_TARGETS:
        raise ValueError(f"Unsupported PEFT method: {peft_cfg['method']}")
    if "target_modules" in raw_yaml_peft:
        peft_cfg["target_modules"] = normalize_target_modules(peft_cfg.get("target_modules"))
    else:
        peft_cfg["target_modules"] = list(METHOD_DEFAULT_TARGETS[peft_cfg["method"]])
    modules_to_save = peft_cfg.get("modules_to_save")
    if isinstance(modules_to_save, str):
        peft_cfg["modules_to_save"] = [modules_to_save]
    elif modules_to_save is None:
        peft_cfg["modules_to_save"] = list(DEFAULT_PEFT_CFG["modules_to_save"])
    return peft_cfg


def build_peft_config(peft_cfg: Dict, cfg: Dict):
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
        adapter_dropout=peft_cfg.get("adapter_dropout", DEFAULT_PEFT_CFG["adapter_dropout"]),
        adapter_scale=peft_cfg.get("adapter_scale", DEFAULT_PEFT_CFG["adapter_scale"]),
        adapter_layernorm_option=peft_cfg.get(
            "adapter_layernorm_option", DEFAULT_PEFT_CFG["adapter_layernorm_option"]
        ),
        fact_rank=peft_cfg.get("fact_rank", DEFAULT_PEFT_CFG["fact_rank"]),
        fact_scale=peft_cfg.get("fact_scale", DEFAULT_PEFT_CFG["fact_scale"]),
        fact_dropout=peft_cfg.get("fact_dropout", DEFAULT_PEFT_CFG["fact_dropout"]),
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
        moe_conv_use_grn=peft_cfg.get("moe_conv_use_grn", DEFAULT_PEFT_CFG["moe_conv_use_grn"]),
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
    )


def build_model(cfg: Dict, disable_peft: bool = False):
    backbone_size = cfg["backbone"].split("_")[-1]
    backbone_version = cfg["backbone"].split("_")[0]
    if backbone_size not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported backbone size: {backbone_size}")

    model_kwargs = {**MODEL_CONFIGS[backbone_size], "nclass": cfg["nclass"]}
    if cfg["model"] == "dpt":
        model = DPT(**model_kwargs, backbone_version=backbone_version)
    elif cfg["model"] == "upernet":
        model = UperNet(**model_kwargs, backbone_version=backbone_version)
    else:
        raise NotImplementedError(f"Unsupported model type: {cfg['model']}")

    pretrained_path = Path("./pretrained") / f"{cfg['backbone']}.pth"
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Backbone pretrained weights not found: {pretrained_path}")

    state_dict = torch.load(str(pretrained_path), map_location="cpu")
    load_result = model.backbone.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        LOGGER.warning(
            "Backbone preload had missing=%d unexpected=%d",
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )

    peft_cfg = None if disable_peft else resolve_peft_cfg(cfg)
    if peft_cfg:
        from peft.tuners.semift import AdaptModel

        if peft_cfg.get("freeze_backbone", True):
            if hasattr(model, "lock_backbone"):
                model.lock_backbone()
            else:
                for param in model.backbone.parameters():
                    param.requires_grad = False
        model = AdaptModel(build_peft_config(peft_cfg, cfg), model)
    return model


def _is_state_dict_like(obj) -> bool:
    return isinstance(obj, dict) and obj and all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()
    )


def extract_state_dict(checkpoint_obj, source: str = "auto") -> Tuple[Dict[str, torch.Tensor], str]:
    if _is_state_dict_like(checkpoint_obj):
        return checkpoint_obj, "state_dict"

    if not isinstance(checkpoint_obj, dict):
        raise ValueError("Unsupported checkpoint format.")

    if source == "model":
        if "model" not in checkpoint_obj:
            raise KeyError("Requested weights-source=model but checkpoint has no 'model'.")
        return checkpoint_obj["model"], "model"
    if source == "model_ema":
        if "model_ema" not in checkpoint_obj:
            raise KeyError("Requested weights-source=model_ema but checkpoint has no 'model_ema'.")
        return checkpoint_obj["model_ema"], "model_ema"

    for key in ("model_ema", "model", "state_dict"):
        if key in checkpoint_obj and _is_state_dict_like(checkpoint_obj[key]):
            return checkpoint_obj[key], key

    if _is_state_dict_like(checkpoint_obj):
        return checkpoint_obj, "state_dict"
    raise ValueError("Could not locate a model state_dict in checkpoint.")


def strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict(state_dict)
    changed = True
    while changed:
        changed = False
        if result and all(key.startswith("module.") for key in result.keys()):
            result = {key[len("module.") :]: value for key, value in result.items()}
            changed = True
    return result


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: Union[str, Path], source: str = "auto") -> str:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict, source_used = extract_state_dict(checkpoint, source=source)
    state_dict = strip_known_prefixes(state_dict)

    model_state_keys = set(model.state_dict().keys())
    overlap = len(model_state_keys & set(state_dict.keys()))
    if overlap == 0:
        raise RuntimeError(
            "Checkpoint has zero overlapping keys with current model. "
            "Please verify config/checkpoint compatibility."
        )

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        LOGGER.warning(
            "Loading %s had %d missing keys (showing up to 10): %s",
            checkpoint_path,
            len(load_result.missing_keys),
            load_result.missing_keys[:10],
        )
    if load_result.unexpected_keys:
        LOGGER.warning(
            "Loading %s had %d unexpected keys (showing up to 10): %s",
            checkpoint_path,
            len(load_result.unexpected_keys),
            load_result.unexpected_keys[:10],
        )
    return source_used


def get_backbone_root(model: torch.nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "backbone"):
        return model.model.backbone
    if hasattr(model, "backbone"):
        return model.backbone
    raise AttributeError("Could not locate backbone on model.")


def get_ignore_index(cfg: Dict) -> int:
    if "ignore_index" in cfg:
        return int(cfg["ignore_index"])
    criterion_kwargs = cfg.get("criterion", {}).get("kwargs", {})
    return int(criterion_kwargs.get("ignore_index", 255))


def sanitize_name(raw: str) -> str:
    name = Path(raw).stem
    name = re.sub(r"[^0-9a-zA-Z._-]+", "_", name)
    return name[:120] if len(name) > 120 else name


def sample_key(dataset_index: int, sample_id: str) -> str:
    return f"{dataset_index:05d}_{sanitize_name(sample_id)}"


def load_image_rgb(path: Union[str, Path]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_manifest(path: Union[str, Path]) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, path: Union[str, Path]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_metrics_csv(records: List[Dict], path: Union[str, Path]):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_metrics_csv(path: Union[str, Path]) -> List[Dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                value = value.strip()
                if value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = int(value)
                    continue
                except ValueError:
                    pass
                try:
                    parsed[key] = float(value)
                    continue
                except ValueError:
                    pass
                parsed[key] = value
            rows.append(parsed)
        return rows


def create_sample_manifest(dataset: ValDataset, split: str, max_samples: int, sample_stride: int) -> Dict:
    entries = []
    count = 0
    stride = max(1, int(sample_stride))
    for dataset_index in range(0, len(dataset), stride):
        sid = dataset.ids[dataset_index]
        entries.append(
            {
                "dataset_index": dataset_index,
                "sample_id": sid,
                "sample_key": sample_key(dataset_index, sid),
            }
        )
        count += 1
        if max_samples > 0 and count >= max_samples:
            break

    return {
        "dataset": dataset.name,
        "split": split,
        "num_entries": len(entries),
        "entries": entries,
        "selection": {
            "max_samples": max_samples,
            "sample_stride": sample_stride,
        },
    }


def validate_manifest_against_dataset(manifest: Dict, dataset: ValDataset, split: str):
    if manifest.get("dataset") != dataset.name:
        raise ValueError(
            f"Manifest dataset {manifest.get('dataset')} does not match current dataset {dataset.name}."
        )
    if manifest.get("split") != split:
        raise ValueError(
            f"Manifest split {manifest.get('split')} does not match requested split {split}."
        )
    for entry in manifest.get("entries", []):
        idx = int(entry["dataset_index"])
        if idx < 0 or idx >= len(dataset):
            raise IndexError(f"Manifest dataset_index out of range: {idx}")
        current_id = dataset.ids[idx]
        if current_id != entry["sample_id"]:
            raise ValueError(
                f"Manifest sample mismatch at index {idx}: {entry['sample_id']} != {current_id}"
            )


def resolve_sample_manifest(dataset: ValDataset, split: str, args, save_dir: Path) -> Tuple[Dict, Path]:
    if args.sample_manifest:
        manifest_path = Path(args.sample_manifest)
        manifest = load_manifest(manifest_path)
    else:
        manifest_path = save_dir / "sample_manifest.json"
        if manifest_path.exists():
            manifest = load_manifest(manifest_path)
        else:
            manifest = create_sample_manifest(dataset, split, args.max_samples, args.sample_stride)
            save_json(manifest, manifest_path)
    validate_manifest_against_dataset(manifest, dataset, split)
    if not manifest_path.exists() or manifest_path.resolve() != (save_dir / "sample_manifest.json").resolve():
        save_json(manifest, save_dir / "sample_manifest.json")
    return manifest, save_dir / "sample_manifest.json"


def manifests_match(manifest_a: Dict, manifest_b: Dict) -> bool:
    return manifest_a.get("dataset") == manifest_b.get("dataset") and manifest_a.get("split") == manifest_b.get("split") and manifest_a.get("entries") == manifest_b.get("entries")


class LayerFeatureCollector:
    def __init__(self, model: torch.nn.Module, token_pool: str = "mean_patch"):
        self.backbone = get_backbone_root(model)
        self.token_pool = token_pool
        self.layer_outputs: List[Optional[torch.Tensor]] = []
        self.layer_labels = [f"block_{idx:02d}" for idx, _ in enumerate(self.backbone.blocks)]
        self.handles = []
        for idx, block in enumerate(self.backbone.blocks):
            self.layer_outputs.append(None)
            self.handles.append(block.register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            self.layer_outputs[idx] = output

        return hook

    def clear(self):
        for idx in range(len(self.layer_outputs)):
            self.layer_outputs[idx] = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def pooled_features(self) -> List[torch.Tensor]:
        pooled = []
        for idx, output in enumerate(self.layer_outputs):
            if output is None:
                raise RuntimeError(f"Layer {idx} did not produce output during forward pass.")
            patch_tokens = extract_patch_tokens(self.backbone, output)
            if self.token_pool == "mean_patch":
                pooled.append(patch_tokens.mean(dim=1).detach().cpu().float())
            else:
                raise NotImplementedError(f"Unsupported token_pool={self.token_pool}")
        return pooled


def extract_patch_tokens(backbone: torch.nn.Module, block_output: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(block_output) or block_output.ndim != 3:
        raise ValueError("Expected block output to be a [B, T, C] tensor.")
    num_prefix = 1
    if hasattr(backbone, "num_register_tokens"):
        num_prefix += int(backbone.num_register_tokens)
    elif hasattr(backbone, "n_storage_tokens"):
        num_prefix += int(backbone.n_storage_tokens)
    if block_output.shape[1] <= num_prefix:
        raise ValueError("Block output contains no patch tokens after removing prefix tokens.")
    return block_output[:, num_prefix:, :]


def prepare_cka_input(x: torch.Tensor, cfg: Dict) -> torch.Tensor:
    eval_mode = cfg.get("eval_mode", "original")
    if eval_mode in {"resize", "slide_window"}:
        crop_size = cfg["crop_size"]
        size = crop_size if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)
    return x


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, cfg: Dict) -> torch.Tensor:
    eval_mode = cfg.get("eval_mode", "original")
    if eval_mode == "slide_window":
        b, _, h, w = x.shape
        final = torch.zeros(b, cfg["nclass"], h, w, device=x.device, dtype=torch.float32)
        size = int(cfg["crop_size"])
        step = 510
        row = 0
        while row <= int(h / step):
            col = 0
            while col <= int(w / step):
                y0 = min(row * step, h - size)
                x0 = min(col * step, w - size)
                sub_input = x[:, :, y0 : min(y0 + size, h), x0 : min(x0 + size, w)]
                logits = model(sub_input)
                final[:, :, y0 : min(y0 + size, h), x0 : min(x0 + size, w)] += logits.float()
                col += 1
            row += 1
        return final

    if eval_mode == "resize":
        resized = prepare_cka_input(x, cfg)
        logits = model(resized)
        return F.interpolate(logits.float(), size=x.shape[-2:], mode="bilinear", align_corners=True)

    return model(x).float()


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def colorize_mask(mask: np.ndarray, dataset_name: str, ignore_index: int) -> np.ndarray:
    cmap = color_map(dataset_name)
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != ignore_index
    if np.any(valid):
        colored[valid] = cmap[mask[valid].astype(np.int64)]
    colored[~valid] = np.array([80, 80, 80], dtype=np.uint8)
    return colored


def error_mask(pred: np.ndarray, target: np.ndarray, ignore_index: int) -> np.ndarray:
    valid = target != ignore_index
    return (pred != target) & valid


def error_map(pred: np.ndarray, target: np.ndarray, ignore_index: int) -> np.ndarray:
    mask = error_mask(pred, target, ignore_index)
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    out[mask] = np.array([255, 0, 0], dtype=np.uint8)
    out[(target == ignore_index)] = np.array([120, 120, 120], dtype=np.uint8)
    return out


def blend_overlay(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    blended = image_uint8.astype(np.float32) * (1.0 - alpha) + mask_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def comparison_delta_map(baseline_pred: np.ndarray, current_pred: np.ndarray, target: np.ndarray, ignore_index: int) -> np.ndarray:
    valid = target != ignore_index
    baseline_correct = (baseline_pred == target) & valid
    current_correct = (current_pred == target) & valid
    out = np.zeros((*target.shape, 3), dtype=np.uint8)
    out[current_correct & (~baseline_correct)] = np.array([0, 220, 0], dtype=np.uint8)
    out[baseline_correct & (~current_correct)] = np.array([220, 0, 0], dtype=np.uint8)
    out[baseline_correct & current_correct] = np.array([40, 40, 40], dtype=np.uint8)
    out[~valid] = np.array([120, 120, 120], dtype=np.uint8)
    return out


def compute_per_image_metrics(pred: np.ndarray, target: np.ndarray, nclass: int, ignore_index: int, dataset_name: str) -> Dict[str, float]:
    intersection, union, target_area = intersectionAndUnion(pred, target, nclass, ignore_index)
    valid = target != ignore_index
    valid_pixels = int(valid.sum())
    correct_pixels = int(((pred == target) & valid).sum())
    error_pixels = int(valid_pixels - correct_pixels)

    union = union.astype(np.float64)
    intersection = intersection.astype(np.float64)
    iou = np.full_like(union, np.nan, dtype=np.float64)
    valid_union = union > 0
    iou[valid_union] = intersection[valid_union] / union[valid_union]
    if dataset_name == "iSAID" and iou.shape[0] > 1:
        miou = float(np.nanmean(iou[1:]))
    else:
        miou = float(np.nanmean(iou))
    pixel_acc = float(correct_pixels / valid_pixels) if valid_pixels > 0 else float("nan")
    error_ratio = float(error_pixels / valid_pixels) if valid_pixels > 0 else float("nan")
    return {
        "miou": miou,
        "pixel_acc": pixel_acc,
        "error_pixels": error_pixels,
        "error_ratio": error_ratio,
        "valid_pixels": valid_pixels,
        "correct_pixels": correct_pixels,
        "target_pixels": int(target_area.sum()),
    }


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("CKA inputs must be 2D arrays [N, C].")
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must have the same number of samples.")
    if x.shape[0] < 2:
        raise ValueError("CKA requires at least 2 samples.")
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    hsic = np.sum(cross * cross)
    x_norm = np.linalg.norm(x.T @ x, ord="fro")
    y_norm = np.linalg.norm(y.T @ y, ord="fro")
    denom = x_norm * y_norm
    if denom <= 0:
        return 0.0
    return float(hsic / denom)


def compute_cka_matrix(features_a: List[np.ndarray], features_b: List[np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(features_a), len(features_b)), dtype=np.float64)
    for i, feat_a in enumerate(features_a):
        for j, feat_b in enumerate(features_b):
            matrix[i, j] = linear_cka(feat_a, feat_b)
    return matrix


def save_heatmap(matrix: np.ndarray, labels_a: Sequence[str], labels_b: Sequence[str], title: str, path: Union[str, Path], dpi: int = 180):
    fig, ax = plt.subplots(figsize=(max(6, len(labels_b) * 0.55), max(5, len(labels_a) * 0.5)))
    im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(labels_b)))
    ax.set_yticks(np.arange(len(labels_a)))
    ax.set_xticklabels(labels_b, rotation=45, ha="right")
    ax.set_yticklabels(labels_a)
    ax.set_xlabel("Layers")
    ax.set_ylabel("Layers")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Linear CKA")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_single_run_artifacts(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, dataset_name: str, ignore_index: int, sample_name: str, save_root: Union[str, Path], save_overlay: bool) -> Dict[str, str]:
    save_root = Path(save_root)
    image_dir = ensure_dir(save_root / "image")
    gt_vis_dir = ensure_dir(save_root / "gt")
    pred_vis_dir = ensure_dir(save_root / "pred")
    error_vis_dir = ensure_dir(save_root / "error")
    arrays_gt_dir = ensure_dir(save_root / "arrays" / "gt")
    arrays_pred_dir = ensure_dir(save_root / "arrays" / "pred")

    gt_vis = colorize_mask(gt, dataset_name, ignore_index)
    pred_vis = colorize_mask(pred, dataset_name, ignore_index)
    err_vis = error_map(pred, gt, ignore_index)

    image_path = image_dir / f"{sample_name}.png"
    gt_vis_path = gt_vis_dir / f"{sample_name}.png"
    pred_vis_path = pred_vis_dir / f"{sample_name}.png"
    error_vis_path = error_vis_dir / f"{sample_name}.png"
    gt_array_path = arrays_gt_dir / f"{sample_name}.npy"
    pred_array_path = arrays_pred_dir / f"{sample_name}.npy"

    plt.imsave(image_path, image)
    plt.imsave(gt_vis_path, gt_vis)
    plt.imsave(pred_vis_path, pred_vis)
    plt.imsave(error_vis_path, err_vis)
    np.save(gt_array_path, gt.astype(np.int32))
    np.save(pred_array_path, pred.astype(np.int32))

    overlay_path = ""
    if save_overlay:
        overlay_dir = ensure_dir(save_root / "overlay")
        overlay_img = blend_overlay(image, pred_vis)
        overlay_path_obj = overlay_dir / f"{sample_name}.png"
        plt.imsave(overlay_path_obj, overlay_img)
        overlay_path = str(overlay_path_obj.relative_to(save_root.parent))

    return {
        "image_path": str(image_path.relative_to(save_root.parent)),
        "gt_vis_path": str(gt_vis_path.relative_to(save_root.parent)),
        "pred_vis_path": str(pred_vis_path.relative_to(save_root.parent)),
        "error_vis_path": str(error_vis_path.relative_to(save_root.parent)),
        "gt_array_path": str(gt_array_path.relative_to(save_root.parent)),
        "pred_array_path": str(pred_array_path.relative_to(save_root.parent)),
        "overlay_path": overlay_path,
    }


def run_single_checkpoint(args):
    save_dir = ensure_dir(args.save_dir)
    cfg = load_yaml(args.config)
    label = args.label or Path(args.checkpoint).stem
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    model = build_model(cfg, disable_peft=args.disable_peft).to(device)
    source_used = load_checkpoint_into_model(model, args.checkpoint, source=args.weights_source)
    model.eval()

    dataset = ValDataset(cfg["dataset"], cfg["data_root"], args.split, ignore_value=get_ignore_index(cfg))
    manifest, manifest_path = resolve_sample_manifest(dataset, args.split, args, save_dir)

    collector = LayerFeatureCollector(model, token_pool=args.token_pool)
    feature_storage: Optional[List[List[torch.Tensor]]] = None
    records: List[Dict] = []
    inference_root = ensure_dir(save_dir / "inference")
    ignore_index = get_ignore_index(cfg)

    for order, entry in enumerate(manifest["entries"]):
        dataset_index = int(entry["dataset_index"])
        expected_sample_id = entry["sample_id"]
        item = dataset[dataset_index]
        image_tensor, mask_tensor, sample_id = item
        if sample_id != expected_sample_id:
            raise ValueError(
                f"Dataset sample mismatch at index {dataset_index}: {sample_id} != {expected_sample_id}"
            )
        key = entry["sample_key"]
        input_tensor = image_tensor.unsqueeze(0).to(device)
        cka_input = prepare_cka_input(input_tensor, cfg)

        collector.clear()
        with torch.no_grad():
            _ = model(cka_input)
        pooled = collector.pooled_features()
        if feature_storage is None:
            feature_storage = [[] for _ in range(len(pooled))]
        for idx, feat in enumerate(pooled):
            feature_storage[idx].append(feat)

        with torch.no_grad():
            logits = predict_logits(model, input_tensor, cfg)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        gt = mask_tensor.numpy().astype(np.int32)
        image_np = denormalize_image(image_tensor)
        metrics = compute_per_image_metrics(pred, gt, cfg["nclass"], ignore_index, cfg["dataset"])
        artifact_paths = save_single_run_artifacts(
            image=image_np,
            gt=gt,
            pred=pred,
            dataset_name=cfg["dataset"],
            ignore_index=ignore_index,
            sample_name=key,
            save_root=inference_root,
            save_overlay=args.save_inference_overlay,
        )
        record = {
            "order": order,
            "dataset_index": dataset_index,
            "sample_id": sample_id,
            "sample_key": key,
            "miou": metrics["miou"],
            "pixel_acc": metrics["pixel_acc"],
            "error_pixels": metrics["error_pixels"],
            "error_ratio": metrics["error_ratio"],
            "valid_pixels": metrics["valid_pixels"],
            **artifact_paths,
        }
        records.append(record)
        LOGGER.info(
            "Processed %d/%d | %s | mIoU=%.4f | error_pixels=%d",
            order + 1,
            len(manifest["entries"]),
            sample_id,
            record["miou"],
            record["error_pixels"],
        )

    collector.close()
    if not records or feature_storage is None:
        raise RuntimeError("No samples processed. Check dataset paths and sample manifest.")

    features = [torch.cat(chunks, dim=0).numpy() for chunks in feature_storage]
    labels = collector.layer_labels
    cka_matrix = compute_cka_matrix(features, features)
    diag_mean = float(np.mean(np.diag(cka_matrix)))

    save_heatmap(
        cka_matrix,
        labels,
        labels,
        f"Self-CKA: {label} ({args.token_pool}, {args.split}, N={len(records)})",
        save_dir / "cka_heatmap.png",
        dpi=args.dpi,
    )
    np.savez_compressed(
        save_dir / "cka_matrix.npz",
        cka=cka_matrix,
        layers=np.array(labels, dtype=object),
    )
    save_metrics_csv(records, save_dir / "sample_metrics.csv")

    metadata = {
        "mode": "single_run",
        "cka_mode": "self",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "label": label,
        "disable_peft": args.disable_peft,
        "weights_source_requested": args.weights_source,
        "weights_source_used": source_used,
        "dataset": cfg["dataset"],
        "nclass": cfg["nclass"],
        "split": args.split,
        "token_pool": args.token_pool,
        "num_processed_samples": len(records),
        "sample_manifest": str(manifest_path.resolve()),
        "diag_mean": diag_mean,
        "device": str(device),
        "save_inference_overlay": bool(args.save_inference_overlay),
    }
    save_json(metadata, save_dir / "cka_metadata.json")
    LOGGER.info("Saved single-run outputs to %s", save_dir)


def discover_run_dirs(runs_root: Union[str, Path]) -> List[Path]:
    runs_root = Path(runs_root)
    results = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "cka_metadata.json").exists() and (child / "sample_metrics.csv").exists() and (child / "sample_manifest.json").exists():
            results.append(child)
    return results


def boxes_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def expand_box(box: Tuple[int, int, int, int], image_h: int, image_w: int, pad: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(image_w, x2 + pad),
        min(image_h, y2 + pad),
    )


def select_zoom_boxes(baseline_pred: np.ndarray, current_pred: np.ndarray, gt: np.ndarray, ignore_index: int, max_boxes: int = 2, pad: int = 24, min_component_area: int = 16) -> List[Tuple[int, int, int, int]]:
    valid = gt != ignore_index
    improve = (current_pred == gt) & (baseline_pred != gt) & valid
    candidate = improve
    if candidate.sum() == 0:
        candidate = (baseline_pred != current_pred) & valid
    if candidate.sum() == 0:
        candidate = ((baseline_pred != gt) | (current_pred != gt)) & valid

    labeled, num_components = ndimage.label(candidate.astype(np.uint8))
    image_h, image_w = gt.shape
    scored_boxes: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for comp_idx in range(1, num_components + 1):
        ys, xs = np.where(labeled == comp_idx)
        if len(xs) == 0:
            continue
        area = len(xs)
        if area < min_component_area and candidate.sum() >= min_component_area:
            continue
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        scored_boxes.append((float(area), expand_box(box, image_h, image_w, pad)))

    if not scored_boxes and candidate.sum() > 0:
        ys, xs = np.where(candidate)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        scored_boxes.append((float(candidate.sum()), expand_box(box, image_h, image_w, pad)))

    if not scored_boxes:
        crop_w = max(32, image_w // 4)
        crop_h = max(32, image_h // 4)
        x1 = max(0, image_w // 2 - crop_w // 2)
        y1 = max(0, image_h // 2 - crop_h // 2)
        return [(x1, y1, min(image_w, x1 + crop_w), min(image_h, y1 + crop_h))]

    scored_boxes.sort(key=lambda item: item[0], reverse=True)
    chosen: List[Tuple[int, int, int, int]] = []
    for _score, box in scored_boxes:
        if all(boxes_iou(box, existing) < 0.3 for existing in chosen):
            chosen.append(box)
        if len(chosen) >= max_boxes:
            break
    return chosen


def crop_by_box(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def render_paper_panel(image: np.ndarray, gt: np.ndarray, baseline_pred: np.ndarray, current_pred: np.ndarray, dataset_name: str, ignore_index: int, baseline_label: str, current_label: str, sample_id: str, summary_record: Dict, boxes: List[Tuple[int, int, int, int]], save_path: Union[str, Path], dpi: int):
    gt_vis = colorize_mask(gt, dataset_name, ignore_index)
    baseline_vis = colorize_mask(baseline_pred, dataset_name, ignore_index)
    current_vis = colorize_mask(current_pred, dataset_name, ignore_index)
    improvement_vis = comparison_delta_map(baseline_pred, current_pred, gt, ignore_index)

    full_panels = [
        ("Image", (image * 255.0).astype(np.uint8)),
        ("GT", gt_vis),
        (baseline_label, baseline_vis),
        (current_label, current_vis),
        ("Improvement", improvement_vis),
    ]
    box_colors = ["lime", "yellow", "cyan", "magenta"]
    n_rows = 1 + max(1, len(boxes))
    fig = plt.figure(figsize=(16, 3.2 * n_rows))
    grid = fig.add_gridspec(n_rows, 5, height_ratios=[1.2] + [1.0] * (n_rows - 1))

    for col, (title, panel_img) in enumerate(full_panels):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(panel_img)
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2.0,
                edgecolor=box_colors[box_idx % len(box_colors)],
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, max(0, y1 - 3), f"Z{box_idx + 1}", color=box_colors[box_idx % len(box_colors)], fontsize=8, bbox=dict(facecolor="black", alpha=0.4, pad=1))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    crop_panels = [
        ("Image Crop", (image * 255.0).astype(np.uint8)),
        ("GT Crop", gt_vis),
        (f"{baseline_label} Crop", baseline_vis),
        (f"{current_label} Crop", current_vis),
        ("Improvement Crop", improvement_vis),
    ]
    if not boxes:
        boxes = [(0, 0, gt.shape[1], gt.shape[0])]

    for row_idx, box in enumerate(boxes, start=1):
        for col, (title, panel_img) in enumerate(crop_panels):
            ax = fig.add_subplot(grid[row_idx, col])
            ax.imshow(crop_by_box(panel_img, box))
            ax.set_title(f"Z{row_idx} {title}", fontsize=9)
            ax.axis("off")

    fig.suptitle(
        (
            f"{sample_id} | {baseline_label}: mIoU={summary_record['baseline_miou']:.4f}, err={summary_record['baseline_error_pixels']} | "
            f"{current_label}: mIoU={summary_record['current_miou']:.4f}, err={summary_record['current_error_pixels']} | "
            f"ΔmIoU={summary_record['miou_gain']:.4f}, ΔErr={summary_record['error_reduction']}"
        ),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compose_overview_pages(panel_paths: List[Path], out_dir: Union[str, Path], per_page: int = 4):
    out_dir = ensure_dir(out_dir)
    if not panel_paths:
        return []
    per_page = max(1, int(per_page))
    output_pages = []
    for page_idx in range(0, len(panel_paths), per_page):
        chunk = panel_paths[page_idx : page_idx + per_page]
        images = [Image.open(path).convert("RGB") for path in chunk]
        max_width = max(img.width for img in images)
        resized = []
        total_height = 0
        for img in images:
            if img.width != max_width:
                new_height = int(round(img.height * max_width / img.width))
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            resized.append(img)
            total_height += img.height
        total_height += 24 * (len(resized) - 1)
        canvas = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
        y = 0
        for img in resized:
            canvas.paste(img, (0, y))
            y += img.height + 24
        page_no = page_idx // per_page + 1
        page_path = out_dir / f"overview_page_{page_no:02d}.png"
        canvas.save(page_path)
        output_pages.append(page_path)
    if len(output_pages) == 1:
        single = out_dir / "overview_topk.png"
        Image.open(output_pages[0]).save(single)
    return output_pages


def build_metrics_lookup(rows: List[Dict]) -> Dict[str, Dict]:
    return {str(row["sample_key"]): row for row in rows}


def compute_summary_records(baseline_rows: List[Dict], current_rows: List[Dict]) -> List[Dict]:
    baseline_lookup = build_metrics_lookup(baseline_rows)
    current_lookup = build_metrics_lookup(current_rows)
    if set(baseline_lookup.keys()) != set(current_lookup.keys()):
        missing = sorted(set(baseline_lookup.keys()) ^ set(current_lookup.keys()))[:10]
        raise ValueError(f"Run sample sets differ. Example differing keys: {missing}")

    records = []
    for key in baseline_lookup:
        base = baseline_lookup[key]
        cur = current_lookup[key]
        records.append(
            {
                "sample_key": key,
                "sample_id": base["sample_id"],
                "dataset_index": int(base["dataset_index"]),
                "baseline_miou": float(base["miou"]),
                "current_miou": float(cur["miou"]),
                "miou_gain": float(cur["miou"] - base["miou"]),
                "baseline_pixel_acc": float(base["pixel_acc"]),
                "current_pixel_acc": float(cur["pixel_acc"]),
                "pixel_acc_gain": float(cur["pixel_acc"] - base["pixel_acc"]),
                "baseline_error_pixels": int(base["error_pixels"]),
                "current_error_pixels": int(cur["error_pixels"]),
                "error_reduction": int(base["error_pixels"] - cur["error_pixels"]),
                "valid_pixels": int(base["valid_pixels"]),
            }
        )
    return records


def sort_summary_records(records: List[Dict], metric: str) -> List[Dict]:
    primary = "miou_gain" if metric == "miou_gain" else "error_reduction"
    return sorted(records, key=lambda row: (row[primary], row["pixel_acc_gain"]), reverse=True)


def render_summary_for_pair(baseline_dir: Path, current_dir: Path, out_dir: Path, args):
    baseline_meta = load_manifest(baseline_dir / "cka_metadata.json")
    current_meta = load_manifest(current_dir / "cka_metadata.json")
    baseline_rows = load_metrics_csv(baseline_dir / "sample_metrics.csv")
    current_rows = load_metrics_csv(current_dir / "sample_metrics.csv")
    summary_records = sort_summary_records(
        compute_summary_records(baseline_rows, current_rows), args.summary_metric
    )
    topk = max(0, int(args.summary_topk))
    selected = summary_records[:topk] if topk > 0 else []

    comparison_dir = ensure_dir(out_dir / "comparison")
    panels_dir = ensure_dir(comparison_dir / "panels")
    save_metrics_csv(selected, comparison_dir / "top_improved.csv")

    ignore_index = int(baseline_meta.get("ignore_index", 255))
    dataset_name = baseline_meta["dataset"]
    baseline_label = baseline_meta["label"]
    current_label = current_meta["label"]

    panel_paths = []
    for rank, row in enumerate(selected, start=1):
        key = row["sample_key"]
        image = load_image_rgb(baseline_dir / "inference" / "image" / f"{key}.png")
        gt = np.load(baseline_dir / "inference" / "arrays" / "gt" / f"{key}.npy")
        baseline_pred = np.load(baseline_dir / "inference" / "arrays" / "pred" / f"{key}.npy")
        current_pred = np.load(current_dir / "inference" / "arrays" / "pred" / f"{key}.npy")
        boxes = select_zoom_boxes(
            baseline_pred,
            current_pred,
            gt,
            ignore_index,
            max_boxes=args.num_zoom_boxes,
            pad=args.zoom_pad,
            min_component_area=args.min_component_area,
        )
        panel_path = panels_dir / f"{rank:03d}_{key}.png"
        render_paper_panel(
            image=image.astype(np.float32) / 255.0,
            gt=gt,
            baseline_pred=baseline_pred,
            current_pred=current_pred,
            dataset_name=dataset_name,
            ignore_index=ignore_index,
            baseline_label=baseline_label,
            current_label=current_label,
            sample_id=row["sample_id"],
            summary_record=row,
            boxes=boxes,
            save_path=panel_path,
            dpi=args.dpi,
        )
        panel_paths.append(panel_path)

    compose_overview_pages(panel_paths, comparison_dir, per_page=args.overview_per_page)
    save_json(
        {
            "baseline_run": baseline_dir.name,
            "current_run": current_dir.name,
            "baseline_label": baseline_label,
            "current_label": current_label,
            "metric": args.summary_metric,
            "summary_topk": topk,
            "num_selected": len(selected),
        },
        comparison_dir / "summary_metadata.json",
    )


def run_summary_mode(args):
    runs_root = Path(args.runs_root)
    run_dirs = discover_run_dirs(runs_root)
    if not run_dirs:
        raise RuntimeError(f"No completed run directories found under {runs_root}")

    baseline_dir = runs_root / args.baseline_run
    if baseline_dir not in run_dirs:
        discovered = [path.name for path in run_dirs]
        raise ValueError(f"Baseline run '{args.baseline_run}' not found. Available: {discovered}")

    baseline_manifest = load_manifest(baseline_dir / "sample_manifest.json")
    out_root = Path(args.summary_output_dir) if args.summary_output_dir else runs_root / "comparison"
    out_root = ensure_dir(out_root)

    compared = []
    for current_dir in run_dirs:
        if current_dir == baseline_dir:
            continue
        current_manifest = load_manifest(current_dir / "sample_manifest.json")
        if not manifests_match(baseline_manifest, current_manifest):
            raise ValueError(
                f"Sample manifest mismatch between baseline '{baseline_dir.name}' and run '{current_dir.name}'."
            )
        pair_dir = ensure_dir(out_root / f"{sanitize_name(baseline_dir.name)}_vs_{sanitize_name(current_dir.name)}")
        render_summary_for_pair(baseline_dir, current_dir, pair_dir, args)
        compared.append(current_dir.name)
        LOGGER.info("Generated summary comparison for %s vs %s", baseline_dir.name, current_dir.name)

    save_json(
        {
            "mode": "summary_only",
            "runs_root": str(runs_root.resolve()),
            "baseline_run": baseline_dir.name,
            "compared_runs": compared,
            "summary_metric": args.summary_metric,
            "summary_topk": args.summary_topk,
        },
        out_root / "summary_index.json",
    )
    LOGGER.info("Saved summary outputs to %s", out_root)


def main():
    args = parse_args()
    validate_args(args)
    setup_logging()
    set_seed(args.seed)

    if args.summary_only:
        run_summary_mode(args)
    else:
        run_single_checkpoint(args)


if __name__ == "__main__":
    main()
