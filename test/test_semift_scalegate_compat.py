import importlib.util
import sys
import types
from pathlib import Path


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

    class SemiFtScaleGate(types.SimpleNamespace):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    moe_mod.SemiFt = SemiFt
    moe_mod.SemiFtScaleGate = SemiFtScaleGate

    sys.modules["peft.utils"] = utils_mod
    sys.modules["peft.tuners.moe"] = moe_mod

    module_name = "peft.tuners.semift_impl_scalegate_compat"
    spec = importlib.util.spec_from_file_location(module_name, SEMIFT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_semift_kwargs_falls_back_for_missing_scalegate_fields():
    module = load_semift_module()
    fake_self = types.SimpleNamespace(
        peft_config=types.SimpleNamespace(
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
            moe_conv_norm_type="layernorm",
        )
    )

    kwargs = module.AdaptModel._semift_kwargs(fake_self)

    assert kwargs["scales"] == [1, 2, 4, 8]
    assert kwargs["conv_gate_temperature"] == 1.0
