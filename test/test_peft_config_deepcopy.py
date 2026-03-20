import copy
import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEMIFT_PATH = REPO_ROOT / "peft" / "tuners" / "semift.py"
CONFIG_PATH = REPO_ROOT / "peft" / "utils" / "config.py"


def load_peft_modules():
    peft_pkg = types.ModuleType("peft")
    peft_pkg.__path__ = []
    utils_pkg = types.ModuleType("peft.utils")
    utils_pkg.__path__ = []
    tuners_pkg = types.ModuleType("peft.tuners")
    tuners_pkg.__path__ = []

    adapters_mod = types.ModuleType("peft.utils.adapters_utils")
    adapters_mod.CONFIG_NAME = "adapter_config.json"
    hub_mod = types.ModuleType("huggingface_hub")
    hub_mod.hf_hub_download = lambda *args, **kwargs: None
    transformers_mod = types.ModuleType("transformers")
    transformers_utils_mod = types.ModuleType("transformers.utils")

    class PushToHubMixin:
        pass

    transformers_utils_mod.PushToHubMixin = PushToHubMixin
    transformers_mod.utils = transformers_utils_mod

    moe_mod = types.ModuleType("peft.tuners.moe")

    class SemiFt(types.SimpleNamespace):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class SemiFtScaleGate(SemiFt):
        pass

    moe_mod.SemiFt = SemiFt
    moe_mod.SemiFtScaleGate = SemiFtScaleGate

    sys.modules["peft"] = peft_pkg
    sys.modules["peft.utils"] = utils_pkg
    sys.modules["peft.tuners"] = tuners_pkg
    sys.modules["peft.utils.adapters_utils"] = adapters_mod
    sys.modules["peft.tuners.moe"] = moe_mod
    sys.modules["huggingface_hub"] = hub_mod
    sys.modules["transformers"] = transformers_mod
    sys.modules["transformers.utils"] = transformers_utils_mod

    config_spec = importlib.util.spec_from_file_location("peft.utils.config", CONFIG_PATH)
    config_module = importlib.util.module_from_spec(config_spec)
    sys.modules["peft.utils.config"] = config_module
    config_spec.loader.exec_module(config_module)

    utils_pkg.PeftConfig = config_module.PeftConfig
    utils_pkg.PeftType = config_module.PeftType

    semift_spec = importlib.util.spec_from_file_location("peft.tuners.semift_impl_deepcopy", SEMIFT_PATH)
    semift_module = importlib.util.module_from_spec(semift_spec)
    sys.modules["peft.tuners.semift_impl_deepcopy"] = semift_module
    semift_spec.loader.exec_module(semift_module)
    return config_module, semift_module


def test_peft_config_to_dict_uses_dataclass_values():
    config_module, _ = load_peft_modules()
    cfg = config_module.PeftConfig(
        base_model_name_or_path="dinov2_small",
        peft_type=config_module.PeftType.LORA,
        inference_mode=True,
    )

    data = cfg.to_dict()

    assert data["base_model_name_or_path"] == "dinov2_small"
    assert data["inference_mode"] is True
    assert cfg.__dict__["base_model_name_or_path"] == "dinov2_small"


def test_semift_config_deepcopy_preserves_scalegate_fields():
    _, semift_module = load_peft_modules()
    cfg = semift_module.SemiFTConfig(
        method="semift_scalegate",
        target_modules=["mlp"],
        moe_expert_scales=[1, 3, 5],
        moe_conv_gate_temperature=0.7,
    )

    cloned = copy.deepcopy(cfg)

    assert cloned.method == "semift_scalegate"
    assert cloned.moe_expert_scales == [1, 3, 5]
    assert cloned.moe_conv_gate_temperature == 0.7


def test_wrapper_deepcopy_preserves_semift_config_state():
    _, semift_module = load_peft_modules()

    class Wrapper:
        def __init__(self):
            self.peft_config = semift_module.SemiFTConfig(
                moe_expert_scales=[2, 4],
                moe_conv_gate_temperature=1.5,
            )

    wrapped = Wrapper()
    cloned = copy.deepcopy(wrapped)

    assert cloned.peft_config.moe_expert_scales == [2, 4]
    assert cloned.peft_config.moe_conv_gate_temperature == 1.5
