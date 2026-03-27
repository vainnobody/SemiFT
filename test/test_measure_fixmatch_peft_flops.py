import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "measure_fixmatch_peft_flops.py"


spec = importlib.util.spec_from_file_location("measure_fixmatch_peft_flops", SCRIPT_PATH)
measure_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = measure_mod
spec.loader.exec_module(measure_mod)


class DummyBaseModel(torch.nn.Module):
    def __init__(self, extra_branch: bool = False):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((8, 8))
        self.head = torch.nn.Linear(4 * 8 * 8, 2)
        self.extra_branch = extra_branch
        if extra_branch:
            self.adapter = torch.nn.Linear(4 * 8 * 8, 2, bias=False)
        for p in self.conv.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False
        if extra_branch:
            for p in self.adapter.parameters():
                p.requires_grad = True

    def forward(self, x, comp_drop=False, feature_perturb=None, need_fp=False):
        x = self.pool(self.conv(x)).flatten(1)
        out = self.head(x)
        if self.extra_branch:
            out = out + self.adapter(x)
        return out


def fake_model_builder(cfg, peft_cfg, load_weights=False):
    method = None if peft_cfg is None else peft_cfg.get("method")
    extra_branch = method not in (None, "bitfit")
    return DummyBaseModel(extra_branch=extra_branch)


def test_get_method_names_matches_peft_registry():
    assert measure_mod.get_method_names() == list(measure_mod.METHOD_DEFAULT_TARGETS.keys())


def test_measure_methods_runs_on_cpu_with_dummy_builder():
    cfg = {
        "crop_size": 64,
        "nclass": 2,
        "model": "dpt",
        "backbone": "dinov3_base",
        "peft": {},
    }
    methods = ["lora", "bitfit", "hydralora"]

    results = measure_mod.measure_methods(
        cfg=cfg,
        methods=methods,
        batch_size=1,
        input_hw=(64, 64),
        device="cpu",
        include_base=True,
        load_backbone_weights=False,
        verbose_breakdown=False,
        strict=True,
        model_builder=fake_model_builder,
    )

    assert [item["method"] for item in results] == ["base", *methods]
    result_map = {item["method"]: item for item in results}

    assert result_map["base"]["status"] == "ok"
    assert result_map["base"]["total_flops"] > 0
    assert result_map["lora"]["total_flops"] > result_map["base"]["total_flops"]
    assert result_map["hydralora"]["total_flops"] > result_map["base"]["total_flops"]
    assert result_map["bitfit"]["total_flops"] == result_map["base"]["total_flops"]
    assert result_map["lora"]["trainable_params"] > 0
    assert result_map["bitfit"]["status"] == "ok"


def test_save_results_supports_json_and_csv(tmp_path):
    results = [
        {
            "method": "base",
            "target_modules": [],
            "total_flops": 123,
            "gflops": 1.23e-7,
            "delta_vs_base_gflops": 0.0,
            "total_params": 10,
            "trainable_params": 1,
            "trainable_ratio": 10.0,
            "status": "ok",
            "error": None,
            "breakdown": None,
        }
    ]
    json_path = tmp_path / "flops.json"
    csv_path = tmp_path / "flops.csv"

    measure_mod.save_results(results, str(json_path))
    measure_mod.save_results(results, str(csv_path))

    assert json_path.exists()
    assert csv_path.exists()
    assert '"method": "base"' in json_path.read_text(encoding="utf-8")
    assert "method,target_modules,total_flops" in csv_path.read_text(encoding="utf-8")
