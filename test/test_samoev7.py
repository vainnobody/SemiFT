import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MOE_PATH = REPO_ROOT / "peft" / "tuners" / "moe.py"


def load_moe_module():
    module_name = "peft.tuners.moe_runtime_test"
    spec = importlib.util.spec_from_file_location(module_name, MOE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_samoev7_forward_exposes_token_level_context_gate_values():
    moe = load_moe_module()
    torch.manual_seed(0)
    adapter = moe.SemiFtSAMoEV7(
        in_features=8,
        out_features=8,
        r=4,
        num_experts=4,
        topk=2,
        num_prefix_tokens=1,
        use_shared_expert=True,
        scales=[1, 2, 4, 8],
        branch_gate_init_bias=-2.0,
    )
    adapter.eval()
    x = torch.randn(2, 5, 8)
    out = adapter(x, hw=(2, 2))

    assert out.shape == x.shape
    assert isinstance(adapter.context_gate, torch.nn.Conv2d)
    assert adapter.context_gate_values is not None
    assert adapter.context_gate_values.shape == (2, 1, 2, 2)


def test_samoev7_context_gate_varies_across_tokens_when_conv_is_configured():
    moe = load_moe_module()
    adapter = moe.SemiFtSAMoEV7(
        in_features=4,
        out_features=4,
        r=2,
        num_experts=4,
        topk=2,
        num_prefix_tokens=1,
        use_shared_expert=False,
        scales=[1, 2, 4, 8],
        branch_gate_init_bias=0.0,
    )
    adapter.eval()
    with torch.no_grad():
        adapter.proj_down.weight.zero_()
        adapter.proj_down.weight[0, 0] = 1.0
        adapter.proj_down.weight[1, 1] = 1.0
        adapter.context_gate.weight.zero_()
        adapter.context_gate.weight[0, 0, 0, 0] = 6.0
        adapter.context_gate.bias.zero_()

    x = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [-4.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]])
    _ = adapter(x, hw=(2, 2))
    gate = adapter.context_gate_values.squeeze(0).squeeze(0)

    assert gate.shape == (2, 2)
    assert torch.unique(gate).numel() > 1


def test_samoev7_without_patch_tokens_clears_context_gate_values():
    moe = load_moe_module()
    adapter = moe.SemiFtSAMoEV7(
        in_features=8,
        out_features=8,
        r=4,
        num_experts=4,
        topk=2,
        num_prefix_tokens=3,
        use_shared_expert=True,
        scales=[1, 2, 4, 8],
    )
    adapter.eval()
    x = torch.randn(2, 3, 8)
    out = adapter(x)

    assert out.shape == x.shape
    assert adapter.context_gate_values is None
