import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util
import types

from unimatchv2_peft import METHOD_DEFAULT_TARGETS, build_peft_config, resolve_peft_cfg
from util.ssl_method_utils import load_backbone_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure forward FLOPs for each PEFT method supported by fixmatch_peft."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", nargs=2, type=int, metavar=("H", "W"), default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--include-base", dest="include_base", action="store_true")
    parser.add_argument("--no-include-base", dest="include_base", action="store_false")
    parser.set_defaults(include_base=True)
    parser.add_argument("--load-backbone-weights", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose-breakdown", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def get_method_names() -> List[str]:
    return list(METHOD_DEFAULT_TARGETS.keys())


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_input_size(cfg: Dict[str, Any], input_size: Optional[Sequence[int]]) -> Tuple[int, int]:
    if input_size is not None:
        return int(input_size[0]), int(input_size[1])
    crop_size = cfg.get("crop_size", 512)
    if isinstance(crop_size, int):
        return crop_size, crop_size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        return int(crop_size[0]), int(crop_size[1])
    raise ValueError(f"Unsupported crop_size format: {crop_size!r}")


def resolve_method_peft_cfg(cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    cfg_copy = copy.deepcopy(cfg)
    args = SimpleNamespace(
        peft_method=method,
        peft_target_modules=None,
        freeze_backbone=None,
    )
    return resolve_peft_cfg(cfg_copy, args)


def load_semift_runtime_module():
    try:
        from peft.tuners.semift import AdaptModel  # noqa: F401
        return
    except Exception:
        pass

    peft_pkg = sys.modules.get("peft")
    if peft_pkg is None:
        peft_pkg = types.ModuleType("peft")
        peft_pkg.__path__ = [str(REPO_ROOT / "peft")]
        sys.modules["peft"] = peft_pkg

    tuners_pkg = sys.modules.get("peft.tuners")
    if tuners_pkg is None:
        tuners_pkg = types.ModuleType("peft.tuners")
        tuners_pkg.__path__ = [str(REPO_ROOT / "peft" / "tuners")]
        sys.modules["peft.tuners"] = tuners_pkg

    utils_mod = types.ModuleType("peft.utils")

    class PeftConfig:
        pass

    class _EnumValue:
        value = "LORA"

    class PeftType:
        LORA = _EnumValue()

    utils_mod.PeftConfig = PeftConfig
    utils_mod.PeftType = PeftType
    sys.modules["peft.utils"] = utils_mod

    moe_spec = importlib.util.spec_from_file_location(
        "peft.tuners.moe", REPO_ROOT / "peft" / "tuners" / "moe.py"
    )
    moe_module = importlib.util.module_from_spec(moe_spec)
    sys.modules["peft.tuners.moe"] = moe_module
    moe_spec.loader.exec_module(moe_module)

    semift_spec = importlib.util.spec_from_file_location(
        "peft.tuners.semift", REPO_ROOT / "peft" / "tuners" / "semift.py"
    )
    semift_module = importlib.util.module_from_spec(semift_spec)
    sys.modules["peft.tuners.semift"] = semift_module
    semift_spec.loader.exec_module(semift_module)


def apply_peft_measurement(model: torch.nn.Module, peft_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> torch.nn.Module:
    load_semift_runtime_module()
    from peft.tuners.semift import AdaptModel

    return AdaptModel(build_peft_config(peft_cfg, cfg), model)


def count_parameter_stats(model: torch.nn.Module) -> Tuple[int, int, float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable_params / total_params if total_params else 0.0
    return total_params, trainable_params, ratio


def build_model_for_measurement(
    cfg: Dict[str, Any],
    peft_cfg: Optional[Dict[str, Any]],
    load_weights: bool = False,
) -> torch.nn.Module:
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
    model_kwargs = {**model_configs[backbone_size], "nclass": cfg["nclass"]}

    if cfg["model"] == "dpt":
        model = DPT(**model_kwargs, backbone_version=backbone_version)
    elif cfg["model"] == "upernet":
        model = UperNet(**model_kwargs, backbone_version=backbone_version)
    else:
        raise NotImplementedError(f"Unsupported model type: {cfg['model']}")

    if load_weights:
        load_backbone_checkpoint(model, cfg)

    if peft_cfg is not None:
        if peft_cfg.get("freeze_backbone", True):
            if hasattr(model, "lock_backbone"):
                model.lock_backbone()
            else:
                for p in model.backbone.parameters():
                    p.requires_grad = False
        model = apply_peft_measurement(model, peft_cfg, cfg)
    return model


def measure_single_model(
    model: torch.nn.Module,
    batch_size: int,
    input_hw: Tuple[int, int],
    device: str,
    verbose_breakdown: bool = False,
) -> Tuple[int, Optional[str]]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model = model.to(device)
    model.eval()
    dummy = torch.randn(batch_size, 3, input_hw[0], input_hw[1], device=device)

    flop_counter = FlopCounterMode(display=False)
    with torch.no_grad():
        with flop_counter:
            _ = model(dummy, comp_drop=False, feature_perturb=None, need_fp=False)

    breakdown = None
    if verbose_breakdown:
        breakdown = flop_counter.get_table(depth=3)
    return int(flop_counter.get_total_flops()), breakdown


def measure_methods(
    cfg: Dict[str, Any],
    methods: Iterable[str],
    batch_size: int,
    input_hw: Tuple[int, int],
    device: str = "cpu",
    include_base: bool = True,
    load_backbone_weights: bool = False,
    verbose_breakdown: bool = False,
    strict: bool = False,
    model_builder=build_model_for_measurement,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    base_flops: Optional[int] = None

    if include_base:
        base_model = model_builder(cfg, peft_cfg=None, load_weights=load_backbone_weights)
        total_params, trainable_params, trainable_ratio = count_parameter_stats(base_model)
        base_flops, breakdown = measure_single_model(
            base_model,
            batch_size=batch_size,
            input_hw=input_hw,
            device=device,
            verbose_breakdown=verbose_breakdown,
        )
        results.append(
            {
                "method": "base",
                "target_modules": [],
                "total_flops": base_flops,
                "gflops": base_flops / 1e9,
                "delta_vs_base_gflops": 0.0,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_ratio": trainable_ratio,
                "status": "ok",
                "error": None,
                "breakdown": breakdown,
            }
        )

    for method in methods:
        try:
            peft_cfg = resolve_method_peft_cfg(cfg, method)
            model = model_builder(cfg, peft_cfg=peft_cfg, load_weights=load_backbone_weights)
            total_params, trainable_params, trainable_ratio = count_parameter_stats(model)
            flops, breakdown = measure_single_model(
                model,
                batch_size=batch_size,
                input_hw=input_hw,
                device=device,
                verbose_breakdown=verbose_breakdown,
            )
            delta_gflops = None if base_flops is None else (flops - base_flops) / 1e9
            results.append(
                {
                    "method": method,
                    "target_modules": peft_cfg.get("target_modules", []),
                    "total_flops": flops,
                    "gflops": flops / 1e9,
                    "delta_vs_base_gflops": delta_gflops,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "trainable_ratio": trainable_ratio,
                    "status": "ok",
                    "error": None,
                    "breakdown": breakdown,
                }
            )
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            results.append(
                {
                    "method": method,
                    "target_modules": [],
                    "total_flops": None,
                    "gflops": None,
                    "delta_vs_base_gflops": None,
                    "total_params": None,
                    "trainable_params": None,
                    "trainable_ratio": None,
                    "status": "error",
                    "error": repr(exc),
                    "breakdown": None,
                }
            )
    return results


def print_results(results: Sequence[Dict[str, Any]]) -> None:
    headers = [
        "method",
        "gflops",
        "delta_vs_base_gflops",
        "trainable_ratio",
        "trainable_params",
        "status",
    ]
    rows = []
    for item in results:
        rows.append(
            [
                item["method"],
                "-" if item["gflops"] is None else f"{item['gflops']:.4f}",
                "-" if item["delta_vs_base_gflops"] is None else f"{item['delta_vs_base_gflops']:.4f}",
                "-" if item["trainable_ratio"] is None else f"{item['trainable_ratio']:.4f}%",
                "-" if item["trainable_params"] is None else f"{item['trainable_params']:,}",
                item["status"],
            ]
        )

    widths = [max(len(str(row[idx])) for row in ([headers] + rows)) for idx in range(len(headers))]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))

    for item in results:
        if item.get("status") == "error":
            print(f"\n[ERROR] {item['method']}: {item['error']}")
        elif item.get("breakdown"):
            print(f"\n[BREAKDOWN] {item['method']}\n{item['breakdown']}")


def save_results(results: Sequence[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for item in results:
        copied = dict(item)
        if isinstance(copied.get("target_modules"), tuple):
            copied["target_modules"] = list(copied["target_modules"])
        serializable.append(copied)

    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if path.suffix.lower() == ".csv":
        fieldnames = [
            "method",
            "target_modules",
            "total_flops",
            "gflops",
            "delta_vs_base_gflops",
            "total_params",
            "trainable_params",
            "trainable_ratio",
            "status",
            "error",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in serializable:
                row = {k: item.get(k) for k in fieldnames}
                row["target_modules"] = json.dumps(row["target_modules"], ensure_ascii=False)
                writer.writerow(row)
        return
    raise ValueError("Unsupported output suffix. Use .json or .csv")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    input_hw = resolve_input_size(cfg, args.input_size)
    methods = args.methods or get_method_names()
    invalid = sorted(set(methods) - set(get_method_names()))
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Supported: {get_method_names()}")

    results = measure_methods(
        cfg=cfg,
        methods=methods,
        batch_size=args.batch_size,
        input_hw=input_hw,
        device=args.device,
        include_base=args.include_base,
        load_backbone_weights=args.load_backbone_weights,
        verbose_breakdown=args.verbose_breakdown,
        strict=args.strict,
    )
    print_results(results)
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
