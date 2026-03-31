import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MOE_PATH = REPO_ROOT / "peft" / "tuners" / "moe.py"


def load_moe_module():
    module_name = "peft.tuners.moe_runtime_test_v9"
    spec = importlib.util.spec_from_file_location(module_name, MOE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRouter(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer("expert_load", torch.zeros(num_experts, dtype=torch.float32))
        if num_experts > 0:
            self.expert_load[0] = 1.0
        self.num_experts = num_experts

    def forward(self, x_2d):
        bsz, _, h, w = x_2d.shape
        n_tokens = h * w
        topk_idx = torch.zeros(bsz, n_tokens, 1, dtype=torch.long, device=x_2d.device)
        topk_weight = torch.ones(bsz, n_tokens, 1, dtype=x_2d.dtype, device=x_2d.device)
        router_probs = torch.zeros(bsz, n_tokens, self.num_experts, dtype=x_2d.dtype, device=x_2d.device)
        router_probs[..., 0] = 1.0
        router_logits = router_probs.clone()
        return topk_idx, topk_weight, {
            "router_logits": router_logits,
            "router_probs": router_probs,
            "selection_scores": router_probs.clone(),
            "expert_bias": self.expert_bias.detach().clone(),
            "expert_load": self.expert_load.detach().clone(),
        }


class _ConstExpert(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x_2d):
        return torch.full_like(x_2d, self.value)


def test_samoev9_forward_exposes_token_level_context_gate_values():
    moe = load_moe_module()
    torch.manual_seed(0)
    adapter = moe.SemiFtSAMoEV9(
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


def test_samoev9_zero_weight_init_matches_bias_prior():
    moe = load_moe_module()
    adapter = moe.SemiFtSAMoEV9(
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
    x_2d = torch.zeros(2, 4, 2, 2)
    sparse_gate = adapter._compute_context_gate(x_2d)
    expected = torch.full_like(sparse_gate, torch.sigmoid(torch.tensor(-2.0)))

    assert torch.allclose(sparse_gate, expected, atol=1e-6)
    assert torch.allclose(1.0 - sparse_gate, torch.ones_like(sparse_gate) - expected, atol=1e-6)


def test_samoev9_uses_complementary_shared_and_sparse_fusion():
    moe = load_moe_module()
    adapter = moe.SemiFtSAMoEV9(
        in_features=2,
        out_features=2,
        r=2,
        num_experts=2,
        topk=1,
        num_prefix_tokens=1,
        use_shared_expert=True,
        scales=[1, 2],
        branch_gate_init_bias=0.0,
    )
    adapter.eval()

    adapter.pre_norm = torch.nn.Identity()
    adapter.input_act = torch.nn.Identity()
    adapter.output_scale = torch.nn.Identity()
    adapter.shared_scale = torch.nn.Identity()
    adapter.moe_scale = torch.nn.Identity()
    adapter.drop_path = torch.nn.Identity()
    adapter.gating_network = _FakeRouter(num_experts=2)
    adapter.shared_expert = _ConstExpert(10.0)

    with torch.no_grad():
        adapter.proj_down.weight.copy_(torch.eye(2))
        adapter.proj_up.weight.copy_(torch.eye(2))

    gate = torch.full((1, 1, 2, 2), 0.25)
    adapter._compute_context_gate = lambda x_2d: gate.to(x_2d.device, x_2d.dtype)
    adapter._sparse_moe_forward = lambda x_2d, topk_idx, topk_weight: torch.full_like(x_2d, 4.0)

    x = torch.zeros(1, 5, 2)
    x[:, 0, :] = 3.0
    out = adapter(x, hw=(2, 2))

    expected_patch = torch.full((1, 4, 2), 8.5)
    assert torch.allclose(out[:, 1:, :], expected_patch, atol=1e-6)
    assert torch.allclose(out[:, :1, :], x[:, :1, :], atol=1e-6)
    assert torch.allclose(adapter.context_gate_values, gate, atol=1e-6)


def test_samoev9_without_patch_tokens_clears_context_gate_values():
    moe = load_moe_module()
    adapter = moe.SemiFtSAMoEV9(
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
