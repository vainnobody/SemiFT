import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
stub_tensorboard = types.ModuleType("torch.utils.tensorboard")


class StubSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass


stub_tensorboard.SummaryWriter = StubSummaryWriter
sys.modules["torch.utils.tensorboard"] = stub_tensorboard

from unimatchv2_peft import build_peft_config, resolve_peft_cfg


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

    class SemiFtScaleGate(SemiFt):
        pass

    class SemiFtSAMoE(SemiFt):
        pass

    class SemiFtSAMoEV4(SemiFt):
        pass

    class SemiFtSAMoEV5(SemiFt):
        pass

    class SemiFtSAMoEV6(SemiFt):
        pass

    class SemiFtSAMoEV7(SemiFt):
        pass

    moe_mod.SemiFt = SemiFt
    moe_mod.SemiFtSAMoE = SemiFtSAMoE
    moe_mod.SemiFtSAMoEV4 = SemiFtSAMoEV4
    moe_mod.SemiFtSAMoEV5 = SemiFtSAMoEV5
    moe_mod.SemiFtSAMoEV6 = SemiFtSAMoEV6
    moe_mod.SemiFtSAMoEV7 = SemiFtSAMoEV7
    moe_mod.SemiFtScaleGate = SemiFtScaleGate

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


def test_fact_methods_default_to_leaf_targets():
    cfg_tt = {"nclass": 6, "peft": {"method": "fact_tt"}}
    cfg_tk = {"nclass": 6, "peft": {"method": "fact_tk"}}

    peft_cfg_tt = resolve_peft_cfg(cfg_tt, make_args())
    peft_cfg_tk = resolve_peft_cfg(cfg_tk, make_args())

    assert peft_cfg_tt["target_modules"] == ["qkv", "proj", "fc1", "fc2"]
    assert peft_cfg_tk["target_modules"] == ["qkv", "proj", "fc1", "fc2"]


def test_semift_router_bias_settings_are_built_from_config():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "semift",
            "target_modules": ["mlp"],
            "moe_router_balance_mode": "deepseek_v3",
            "moe_router_bias_update_speed": 0.002,
            "moe_router_bias_clip": 0.1,
            "moe_router_aux_loss_coef": 0.0,
            "moe_router_z_loss_coef": 0.0,
        },
    }
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    config = build_peft_config(peft_cfg, cfg)

    assert config.method == "semift"
    assert config.moe_router_balance_mode == "deepseek_v3"
    assert abs(config.moe_router_bias_update_speed - 0.002) < 1e-8
    assert abs(config.moe_router_bias_clip - 0.1) < 1e-8
    assert config.moe_router_aux_loss_coef == 0.0
    assert config.moe_router_z_loss_coef == 0.0


def test_conv_lora_config_fields_are_built_from_config():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "conv_lora",
            "target_modules": ["qkv"],
            "conv_lora_num_experts": 3,
            "conv_lora_topk": 1,
            "conv_lora_kernel_size": 5,
            "conv_lora_dropout": 0.2,
        },
    }
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    config = build_peft_config(peft_cfg, cfg)

    assert config.method == "conv_lora"
    assert config.conv_lora_num_experts == 3
    assert config.conv_lora_topk == 1
    assert config.conv_lora_kernel_size == 5
    assert abs(config.conv_lora_dropout - 0.2) < 1e-8


def test_semift_samoe_config_builds_drop_path_rate():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "semift_samoe",
            "target_modules": ["mlp"],
            "moe_expert_drop_path_rate": 0.15,
        },
    }
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    config = build_peft_config(peft_cfg, cfg)
    assert config.method == "semift_samoe"
    assert abs(config.moe_expert_drop_path_rate - 0.15) < 1e-8


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


def test_adaptmodel_wraps_fact_tt_leaf_modules():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="fact_tt", target_modules=["mlp"], fact_rank=4)
    adapted = semift.AdaptModel(cfg, model)
    assert isinstance(adapted.model.block.mlp.fc1, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.fc2, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.fc1.adapter, semift.FactTTAdapter)
    assert isinstance(adapted.model.block.mlp.fc2.adapter, semift.FactTTAdapter)


def test_adaptmodel_wraps_fact_tk_leaf_modules():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="fact_tk", target_modules=["mlp"], fact_rank=4)
    adapted = semift.AdaptModel(cfg, model)
    assert isinstance(adapted.model.block.mlp.fc1, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.fc2, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.fc1.adapter, semift.FactTKAdapter)
    assert isinstance(adapted.model.block.mlp.fc2.adapter, semift.FactTKAdapter)


def test_adaptmodel_infers_prefix_tokens_for_conv_lora():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="conv_lora", target_modules=["qkv"], r=4)
    adapted = semift.AdaptModel(cfg, model)

    assert isinstance(adapted.model.block.attn.qkv, semift.WarpBlock)
    assert isinstance(adapted.model.block.attn.qkv.adapter, semift.ConvLora)
    assert adapted.model.block.attn.qkv.adapter.num_prefix_tokens == 1


def test_conv_lora_uses_spatial_branch_when_prefix_tokens_are_valid():
    semift = load_semift_module()
    import torch

    adapter = semift.ConvLora(
        in_features=4,
        out_features=4,
        r=2,
        lora_alpha=2,
        dropout=0.0,
        kernel_size=1,
        num_experts=2,
        topk=1,
        num_prefix_tokens=1,
    )
    with torch.no_grad():
        adapter.lora.lora_A.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
        adapter.gate.proj.weight.zero_()
        adapter.gate.proj.weight[0, 0] = 1.0
        adapter.experts[0][0].weight.fill_(1.0)
        adapter.experts[1][0].weight.zero_()
        adapter.spatial_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            )
        )

    x = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    out = adapter(x)

    assert out.shape == x.shape
    assert torch.allclose(out[:, :1], torch.zeros_like(out[:, :1]))
    assert not torch.allclose(out[:, 1:], torch.zeros_like(out[:, 1:]))


def test_conv_lora_falls_back_to_plain_lora_when_tokens_are_not_square():
    semift = load_semift_module()
    import torch

    adapter = semift.ConvLora(
        in_features=4,
        out_features=4,
        r=2,
        lora_alpha=2,
        dropout=0.0,
        kernel_size=1,
        num_experts=2,
        topk=1,
        num_prefix_tokens=1,
    )
    x = torch.randn(2, 6, 4)
    out = adapter(x)
    ref = adapter.lora(x)

    assert torch.allclose(out, ref)


def test_hydralora_uses_linear_router_like_official_design():
    semift = load_semift_module()
    import torch.nn as nn

    adapter = semift.HydraLora(
        in_features=8,
        out_features=4,
        r=2,
        num_branches=3,
        router_hidden=16,
        router_dropout=0.2,
        lora_alpha=8,
        dropout=0.1,
    )

    assert isinstance(adapter.shared_A, nn.Linear)
    assert isinstance(adapter.router, nn.Linear)
    assert len(adapter.branches) == 3


def test_hydralora_matches_shared_a_multi_b_weighted_sum():
    semift = load_semift_module()
    import torch

    adapter = semift.HydraLora(
        in_features=2,
        out_features=1,
        r=1,
        num_branches=2,
        router_hidden=4,
        router_dropout=0.0,
        lora_alpha=1,
        dropout=0.0,
    )
    with torch.no_grad():
        adapter.shared_A.weight.copy_(torch.tensor([[1.0, 0.0]]))
        adapter.router.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        adapter.branches[0].weight.copy_(torch.tensor([[2.0]]))
        adapter.branches[1].weight.copy_(torch.tensor([[4.0]]))

    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    out = adapter(x)

    weights = torch.softmax(adapter.router(x), dim=-1)
    hidden = adapter.shared_A(x)
    expected = (
        adapter.branches[0](hidden) * weights[..., :1]
        + adapter.branches[1](hidden) * weights[..., 1:]
    ) * adapter.scaling

    assert torch.allclose(out, expected, atol=1e-6)


def test_adaptmodel_wraps_adaptformer_block():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="adaptformer", target_modules=["mlp"], adapter_dim=4)
    adapted = semift.AdaptModel(cfg, model)
    assert isinstance(adapted.model.block, semift.AdaptFormerBlockWrapper)
    assert isinstance(adapted.model.block.adapter, semift.AdapterFormer)


def test_adaptformer_block_wrapper_keeps_parallel_residual_path():
    semift = load_semift_module()
    import torch
    import torch.nn as nn

    class ToyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.Identity()
            self.attn = nn.Sequential(nn.Identity(), nn.Linear(8, 8, bias=False))
            self.ls1 = nn.Identity()
            self.drop_path1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.drop_path2 = nn.Identity()
            self.sample_drop_ratio = 0.0
            self.mlp = nn.Linear(8, 8, bias=False)
            self.ls2 = nn.Identity()
            with torch.no_grad():
                self.attn[1].weight.zero_()
                self.mlp.weight.copy_(torch.eye(8))

        def forward(self, x):
            return x + self.mlp(x)

    block = ToyBlock()
    adapter = semift.AdapterFormer(8, 8, r=4, dropout=0.0, scale=0.1, layernorm_option="none")
    wrapped = semift.AdaptFormerBlockWrapper(block, adapter)
    x = torch.randn(2, 3, 8)
    out = wrapped(x)

    assert out.shape == x.shape
    assert torch.allclose(out, x + block.mlp(x), atol=1e-6)


def test_adaptformer_block_wrapper_routes_single_positional_arg_as_rope():
    semift = load_semift_module()
    import torch
    import torch.nn as nn

    class RopeAwareAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_rope = None

        def forward(self, x, attn_bias=None, rope=None):
            assert attn_bias is None
            self.last_rope = rope
            return torch.zeros_like(x)

    class ToyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.Identity()
            self.attn = RopeAwareAttn()
            self.norm2 = nn.Identity()
            self.mlp = nn.Linear(8, 8, bias=False)
            self.sample_drop_ratio = 0.0
            with torch.no_grad():
                self.mlp.weight.zero_()

        def forward(self, x):
            return x

    block = ToyBlock()
    adapter = semift.AdapterFormer(8, 8, r=4, dropout=0.0, scale=0.1, layernorm_option="none")
    wrapped = semift.AdaptFormerBlockWrapper(block, adapter)
    x = torch.randn(2, 3, 8)
    rope = torch.randn(3, 4)
    wrapped(x, rope)

    assert block.attn.last_rope is rope


def test_adaptmodel_falls_back_to_whole_patch_embed_wrapper_without_proj_or_norm():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="ssf", target_modules=["patch_embed"])
    adapted = semift.AdaptModel(cfg, model)

    assert hasattr(adapted.model.patch_embed, "proj") is False
    assert isinstance(adapted.model.patch_embed, semift.SsfWrapper)


def test_adaptmodel_wraps_ssf_patch_embed_proj_and_norm_modules():
    semift = load_semift_module()
    import torch.nn as nn

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, 8, kernel_size=2, stride=2)
            self.norm = nn.LayerNorm(8)

        def forward(self, x):
            x = self.proj(x)
            x = x.flatten(2).transpose(1, 2)
            return self.norm(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = PatchEmbed()

        def forward(self, x):
            return self.patch_embed(x)

    model = Model()
    cfg = semift.SemiFTConfig(method="ssf", target_modules=["patch_embed"])
    adapted = semift.AdaptModel(cfg, model)

    assert isinstance(adapted.model.patch_embed.proj, semift.SsfWrapper)
    assert isinstance(adapted.model.patch_embed.norm, semift.SsfWrapper)
    assert adapted.model.patch_embed.proj.base_layer.weight.requires_grad is False
    assert adapted.model.patch_embed.norm.base_layer.weight.requires_grad is False


def test_adaptmodel_ssf_default_targets_do_not_rewrap_patch_embed_children():
    semift = load_semift_module()
    import torch.nn as nn

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, 8, kernel_size=2, stride=2)
            self.norm = nn.LayerNorm(8)

        def forward(self, x):
            x = self.proj(x)
            x = x.flatten(2).transpose(1, 2)
            return self.norm(x)

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
            self.patch_embed = PatchEmbed()
            self.block = Block()

        def forward(self, x):
            return self.block(self.patch_embed(x))

    cfg = semift.SemiFTConfig(method="ssf", target_modules=semift.METHOD_DEFAULT_TARGETS["ssf"])
    adapted = semift.AdaptModel(cfg, Model())

    assert isinstance(adapted.model.patch_embed.proj, semift.SsfWrapper)
    assert isinstance(adapted.model.patch_embed.norm, semift.SsfWrapper)


def test_bitfit_only_enables_biases_for_target_module():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="bitfit", target_modules=["mlp"])
    adapted = semift.AdaptModel(cfg, model)
    assert adapted.model.block.mlp.fc1.weight.requires_grad is False
    assert adapted.model.block.mlp.fc2.weight.requires_grad is False
    assert adapted.model.block.mlp.fc1.bias.requires_grad is True
    assert adapted.model.block.mlp.fc2.bias.requires_grad is True


def test_bitfit_keeps_head_fully_trainable():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="bitfit", target_modules=["head"])
    adapted = semift.AdaptModel(cfg, model)

    assert adapted.model.head.weight.requires_grad is True
    assert adapted.model.head.bias.requires_grad is True


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


def test_warpblock_exposes_base_layer_linear_attributes():
    semift = load_semift_module()
    import torch.nn as nn

    base = nn.Linear(8, 16)
    wrapped = semift.WarpBlock(base, semift.Lora(8, 16, r=4, lora_alpha=8))

    assert wrapped.in_features == 8
    assert wrapped.out_features == 16
    assert wrapped.weight is base.weight


def test_semift_scalegate_router_settings_are_built_from_config():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "semift_scalegate",
            "target_modules": ["mlp"],
            "moe_expert_scales": [1, 2, 4, 8],
            "moe_conv_gate_temperature": 0.7,
        },
    }
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    config = build_peft_config(peft_cfg, cfg)

    assert config.method == "semift_scalegate"
    assert config.moe_expert_scales == [1, 2, 4, 8]
    assert abs(config.moe_conv_gate_temperature - 0.7) < 1e-8


def test_adaptmodel_wraps_semift_scalegate_block():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="semift_scalegate", target_modules=["mlp"], r=4)
    adapted = semift.AdaptModel(cfg, model)
    assert isinstance(adapted.model.block.mlp, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.adapter, semift.SemiFtScaleGate)


def test_adaptmodel_wraps_semift_samoe_block():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="semift_samoe", target_modules=["mlp"], r=4, moe_num_prefix_tokens=-1)
    adapted = semift.AdaptModel(cfg, model)
    assert isinstance(adapted.model.block.mlp, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.adapter, semift.SemiFtSAMoE)


def test_adaptmodel_wraps_samoev4_block():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="samoev4", target_modules=["mlp"], r=4, moe_num_prefix_tokens=-1)
    adapted = semift.AdaptModel(cfg, model)
    assert isinstance(adapted.model.block.mlp, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.adapter, semift.SemiFtSAMoEV4)


def test_adaptmodel_wraps_samoev5_block():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="samoev5", target_modules=["mlp"], r=4, moe_num_prefix_tokens=-1)
    adapted = semift.AdaptModel(cfg, model)
    assert isinstance(adapted.model.block.mlp, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.adapter, semift.SemiFtSAMoEV5)


def test_build_peft_config_passes_scalegate_fields_to_current_config():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "semift_scalegate",
            "target_modules": ["mlp"],
            "moe_expert_scales": [1, 3, 5],
            "moe_conv_gate_temperature": 0.7,
        },
    }
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    config = build_peft_config(peft_cfg, cfg)

    assert config.method == "semift_scalegate"
    assert config.moe_expert_scales == [1, 3, 5]
    assert abs(config.moe_conv_gate_temperature - 0.7) < 1e-8


def test_build_peft_config_supports_semift_samoe():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "semift_samoe",
            "target_modules": ["mlp"],
            "moe_expert_scales": [1, 2, 4, 8],
            "moe_layerscale_init": 1e-5,
            "moe_num_prefix_tokens": -1,
        },
    }
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    config = build_peft_config(peft_cfg, cfg)

    assert config.method == "semift_samoe"
    assert config.moe_expert_scales == [1, 2, 4, 8]
    assert abs(config.moe_layerscale_init - 1e-5) < 1e-8
    assert config.moe_num_prefix_tokens == -1


def test_build_peft_config_supports_samoev4():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "samoev4",
            "target_modules": ["mlp"],
            "moe_expert_scales": [1, 2, 4, 8],
            "moe_layerscale_init": 1e-5,
            "moe_num_prefix_tokens": -1,
        },
    }
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    config = build_peft_config(peft_cfg, cfg)

    assert config.method == "samoev4"
    assert config.moe_expert_scales == [1, 2, 4, 8]
    assert abs(config.moe_layerscale_init - 1e-5) < 1e-8
    assert config.moe_num_prefix_tokens == -1


def test_build_peft_config_supports_samoev5():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "samoev5",
            "target_modules": ["mlp"],
            "moe_expert_scales": [1, 2, 4, 8],
            "moe_layerscale_init": 1e-5,
            "moe_num_prefix_tokens": -1,
            "moe_branch_gate_init_bias": -1.25,
        },
    }
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    config = build_peft_config(peft_cfg, cfg)

    assert config.method == "samoev5"
    assert config.moe_expert_scales == [1, 2, 4, 8]
    assert abs(config.moe_layerscale_init - 1e-5) < 1e-8
    assert config.moe_num_prefix_tokens == -1
    assert abs(config.moe_branch_gate_init_bias + 1.25) < 1e-8


def test_adaptmodel_wraps_samoev7_block():
    semift = load_semift_module()
    model = build_dummy_block(semift.torch)
    cfg = semift.SemiFTConfig(method="samoev7", target_modules=["mlp"], r=4, moe_num_prefix_tokens=-1)
    adapted = semift.AdaptModel(cfg, model)
    assert isinstance(adapted.model.block.mlp, semift.WarpBlock)
    assert isinstance(adapted.model.block.mlp.adapter, semift.SemiFtSAMoEV7)


def test_build_peft_config_supports_samoev7():
    cfg = {
        "nclass": 6,
        "peft": {
            "method": "samoev7",
            "target_modules": ["mlp"],
            "moe_expert_scales": [1, 2, 4, 8],
            "moe_layerscale_init": 1e-5,
            "moe_num_prefix_tokens": -1,
            "moe_branch_gate_init_bias": -1.25,
        },
    }
    peft_cfg = resolve_peft_cfg(cfg, make_args())
    config = build_peft_config(peft_cfg, cfg)

    assert config.method == "samoev7"
    assert config.moe_expert_scales == [1, 2, 4, 8]
    assert abs(config.moe_layerscale_init - 1e-5) < 1e-8
    assert config.moe_num_prefix_tokens == -1
    assert abs(config.moe_branch_gate_init_bias + 1.25) < 1e-8


def test_semift_kwargs_supports_samoev7_branch_gate_bias():
    module = load_semift_module()
    fake_self = types.SimpleNamespace(
        peft_config=types.SimpleNamespace(
            method="samoev7",
            r=32,
            moe_num_experts=4,
            moe_topk=2,
            moe_router_balance_mode="deepseek_v3",
            moe_router_bias_update_speed=1e-3,
            moe_router_bias_clip=0.05,
            moe_router_jitter_noise=1e-2,
            moe_num_prefix_tokens=5,
            moe_use_shared_expert=True,
            moe_conv_hidden_ratio=2.0,
            moe_conv_kernel_size=3,
            moe_conv_context_kernel_size=5,
            moe_conv_use_grn=True,
            moe_conv_norm_type="groupnorm",
            moe_expert_scales=[1, 2, 4, 8],
            moe_conv_gate_temperature=1.0,
            moe_layerscale_init=1e-5,
            moe_expert_drop_path_rate=0.1,
            moe_branch_gate_init_bias=-1.5,
        )
    )
    kwargs = module.AdaptModel._semift_kwargs(fake_self)
    assert kwargs["layerscale_init"] == 1e-5
    assert kwargs["drop_path_rate"] == 0.1
    assert kwargs["branch_gate_init_bias"] == -1.5
    assert "conv_hidden_ratio" not in kwargs
    assert "conv_context_kernel_size" not in kwargs
