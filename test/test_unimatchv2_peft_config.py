import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


stub_peft = types.ModuleType("peft")
stub_peft.__path__ = []
stub_peft_tuners = types.ModuleType("peft.tuners")
stub_peft_tuners.__path__ = []
stub_peft_semift = types.ModuleType("peft.tuners.semift")


class StubSemiFTConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class StubAdaptModel:
    def __init__(self, config, model):
        self.config = config
        self.model = model


stub_peft_semift.SemiFTConfig = StubSemiFTConfig
stub_peft_semift.AdaptModel = StubAdaptModel

sys.modules.setdefault("peft", stub_peft)
sys.modules.setdefault("peft.tuners", stub_peft_tuners)
sys.modules["peft.tuners.semift"] = stub_peft_semift

from unimatchv2_peft import build_peft_config, resolve_peft_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]
SEMIFT_PATH = REPO_ROOT / "peft" / "tuners" / "semift.py"


def load_semift_module():
    utils_mod = types.ModuleType("peft.utils")

    class PeftConfig:
        pass

    class _EnumValue:
        value = "LORA"

    class PeftType:
        LORA = _EnumValue()

    utils_mod.PeftConfig = PeftConfig
    utils_mod.PeftType = PeftType

    moe_mod = types.ModuleType("peft.tuners.moe")

    class SemiFt(types.SimpleNamespace):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, x):
            return x

    moe_mod.SemiFt = SemiFt

    sys.modules["peft.utils"] = utils_mod
    sys.modules["peft.tuners.moe"] = moe_mod

    module_name = "peft.tuners.semift_impl"
    spec = importlib.util.spec_from_file_location(module_name, SEMIFT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_args(**kwargs):
    defaults = {
        "peft_method": None,
        "peft_target_modules": None,
        "freeze_backbone": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_resolve_peft_cfg_from_yaml():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "lora",
            "target_modules": ["attn", "mlp"],
            "freeze_backbone": False,
            "modules_to_save": "head",
            "r": 16,
            "lora_alpha": 48,
            "lora_dropout": 0.2,
        },
    }
    peft_cfg = resolve_peft_cfg(cfg, make_args())

    assert peft_cfg["method"] == "lora"
    assert peft_cfg["target_modules"] == ["attn", "mlp"]
    assert peft_cfg["freeze_backbone"] is False
    assert peft_cfg["modules_to_save"] == ["head"]

    config = build_peft_config(peft_cfg, cfg)
    assert config.method == "lora"
    assert config.r == 16
    assert config.lora_alpha == 48
    assert abs(config.lora_dropout - 0.2) < 1e-8


def test_cli_overrides_yaml_peft_cfg():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "semift",
            "target_modules": ["mlp"],
            "freeze_backbone": True,
        },
    }
    peft_cfg = resolve_peft_cfg(
        cfg,
        make_args(
            peft_method="lora",
            peft_target_modules=["attn"],
            freeze_backbone=False,
        ),
    )

    assert peft_cfg["method"] == "lora"
    assert peft_cfg["target_modules"] == ["attn"]
    assert peft_cfg["freeze_backbone"] is False


def test_method_default_targets_are_selected_when_not_provided():
    cfg = {"nclass": 6, "peft": {"method": "hydralora"}}
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    assert peft_cfg["target_modules"] == ["qkv", "proj", "fc1", "fc2"]


class DummyAttention(types.SimpleNamespace):
    pass


class DummyMLP(types.SimpleNamespace):
    pass


def build_dummy_block(torch_mod):
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24)
            self.proj = nn.Linear(8, 8)

        def forward(self, x):
            return self.proj(self.qkv(x).chunk(3, dim=-1)[0])

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 8)

        def forward(self, x):
            return self.fc2(self.fc1(x).relu())

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(8)
            self.attn = Attn()
            self.norm2 = nn.LayerNorm(8)
            self.mlp = Mlp()

        def forward(self, x):
            x = self.attn(self.norm1(x))
            return self.mlp(self.norm2(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(8, 8)
            self.block = Block()
            self.head = nn.Linear(8, 2)

        def forward(self, x):
            return self.head(self.block(self.patch_embed(x)))

    return Model()


def test_adaptmodel_wraps_lora_leaf_modules():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="lora", target_modules=["mlp"], r=4, lora_alpha=8)
    adapted = semift.AdaptModel(cfg, model)
    assert isinstance(adapted.model.block.mlp.fc1, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.fc2, semift.WarpBlock)


def test_adaptmodel_wraps_adaptformer_block():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="adaptformer", target_modules=["mlp"], adapter_dim=4)
    adapted = semift.AdaptModel(cfg, model)
    assert isinstance(adapted.model.block.mlp, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.adapter, semift.AdapterFormer)


def test_bitfit_only_enables_biases_for_target_module():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="bitfit", target_modules=["mlp"])
    adapted = semift.AdaptModel(cfg, model)
    assert adapted.model.block.mlp.fc1.weight.requires_grad is False
    assert adapted.model.block.mlp.fc2.weight.requires_grad is False
    assert adapted.model.block.mlp.fc1.bias.requires_grad is True
    assert adapted.model.block.mlp.fc2.bias.requires_grad is True


def test_adaptmodel_skips_non_linear_proj_suffix_matches():
    semift = load_semift_module()
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24)
            self.proj = nn.Linear(8, 8)

        def forward(self, x):
            return self.proj(self.qkv(x).chunk(3, dim=-1)[0])

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Conv2d(8, 8, kernel_size=1), nn.ReLU())

        def forward(self, x):
            return self.proj(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()
            self.decoder = Decoder()

        def forward(self, x):
            return self.attn(x)

    model = Model()
    decoder_proj = model.decoder.proj
    cfg = semift.SemiFTConfig(method="lora", target_modules=["proj"], r=4, lora_alpha=8)
    adapted = semift.AdaptModel(cfg, model)

    assert isinstance(adapted.model.attn.proj, semift.WarpBlock)
    assert adapted.model.decoder.proj is decoder_proj
