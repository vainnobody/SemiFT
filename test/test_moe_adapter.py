import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TUNERS_DIR = REPO_ROOT / "peft" / "tuners"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


moe = _load_module("local_moe", TUNERS_DIR / "moe.py")


def test_conv_expert_forward_with_explicit_hw():
    expert = moe.ConvExpert(r=8, hidden_ratio=2.0, kernel_size=3, context_kernel_size=5)
    x = torch.randn(2, 15, 8)
    y = expert(x, hw=(3, 5))
    assert y.shape == x.shape


def test_semift_forward_shape_and_aux_loss():
    module = moe.SemiFt(
        in_features=16,
        out_features=16,
        r=8,
        num_experts=4,
        topk=2,
        num_prefix_tokens=5,
    )
    module.train()

    x = torch.randn(2, 14, 16, requires_grad=True)
    y = module(x)

    assert y.shape == x.shape
    assert module.aux_loss is not None
    assert module.aux_loss.item() == 0.0
    assert module.router_aux_loss.item() == 0.0
    assert module.router_z_loss.item() == 0.0
    assert module.last_hw == (3, 3)
    assert module.selection_scores.shape == (2, 9, 4)
    assert module.expert_bias.shape == (4,)
    assert module.expert_load.shape == (4,)

    probs = torch.softmax(module.router_logits, dim=-1)
    assert torch.isfinite(probs).all()
    assert torch.isfinite(module.selection_scores).all()
    assert torch.isfinite(module.expert_bias).all()
    assert torch.isfinite(module.expert_load).all()
    assert torch.allclose(module.expert_load.sum(), torch.tensor(1.0), atol=1e-6)

    loss = y.sum()
    loss.backward()

    assert module.proj_down.weight.grad is not None
    assert module.gating_network.gate_proj.weight.grad is not None
    assert module.experts[0].pw_expand.weight.grad is not None


def test_semift_handles_non_square_patch_tokens_with_hw():
    module = moe.SemiFt(
        in_features=12,
        out_features=12,
        r=6,
        num_experts=3,
        topk=2,
        num_prefix_tokens=5,
    )
    module.eval()

    x = torch.randn(1, 18, 12)  # 13 patch tokens after removing 5 prefix tokens
    y = module(x, hw=(1, 13))

    assert y.shape == x.shape
    assert module.router_logits.shape == (1, 13, 3)
    assert module.selection_scores.shape == (1, 13, 3)
    assert module.last_hw == (1, 13)


def test_semift_non_square_patch_tokens_without_hw_raises():
    module = moe.SemiFt(
        in_features=12,
        out_features=12,
        r=6,
        num_experts=3,
        topk=2,
        num_prefix_tokens=5,
    )
    x = torch.randn(1, 18, 12)
    try:
        module(x)
    except ValueError as exc:
        assert "Please pass hw" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-square patch tokens without hw")


def test_semift_zero_patch_tokens_is_safe():
    module = moe.SemiFt(
        in_features=10,
        out_features=10,
        r=4,
        num_experts=2,
        topk=1,
        num_prefix_tokens=5,
    )
    x = torch.randn(2, 5, 10)
    y = module(x)
    assert y.shape == x.shape
    assert module.aux_loss.item() == 0.0


def test_gating_weights_are_normalized():
    gate = moe.GatingNetwork(r=8, num_experts=4, topk=2)
    x = torch.randn(2, 7, 8)
    idx, weight, stats = gate(x)

    assert idx.shape == (2, 7, 2)
    assert weight.shape == (2, 7, 2)
    assert torch.allclose(weight.sum(dim=-1), torch.ones_like(weight[..., 0]), atol=1e-5)
    assert "aux_loss" in stats and "z_loss" in stats and "router_probs" in stats
    assert "selection_scores" in stats and "expert_bias" in stats and "expert_load" in stats
    assert stats["aux_loss"].item() == 0.0
    assert stats["z_loss"].item() == 0.0


def test_gating_updates_bias_only_in_train_mode():
    gate = moe.GatingNetwork(
        r=8,
        num_experts=4,
        topk=2,
        balance_mode="deepseek_v3",
        bias_update_speed=1e-2,
        bias_clip=0.05,
    )
    with torch.no_grad():
        gate.gate_proj.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
    x = torch.arange(8, dtype=torch.float32).view(1, 1, 8).repeat(2, 7, 1)

    gate.eval()
    bias_before_eval = gate.expert_bias.clone()
    gate(x)
    assert torch.allclose(gate.expert_bias, bias_before_eval)

    gate.train()
    bias_before_train = gate.expert_bias.clone()
    for _ in range(4):
        gate(x)
    assert not torch.allclose(gate.expert_bias, bias_before_train)
    assert torch.all(gate.expert_bias.abs() <= 0.05 + 1e-8)


def test_gating_eval_skips_distributed_expert_load_sync(monkeypatch):
    gate = moe.GatingNetwork(r=8, num_experts=4, topk=2)
    x = torch.randn(2, 7, 8)

    monkeypatch.setattr(moe.dist, "is_available", lambda: True)
    monkeypatch.setattr(moe.dist, "is_initialized", lambda: True)

    def fail_all_reduce(*args, **kwargs):
        raise AssertionError("all_reduce should not run during eval forward")

    monkeypatch.setattr(moe.dist, "all_reduce", fail_all_reduce)

    gate.eval()
    gate(x)


def test_scale_context_router_skips_all_reduce_for_world_size_one(monkeypatch):
    router = moe.ScaleContextRouter(r=8, num_experts=4, topk=2, scales=[1, 2, 4, 8])
    x = torch.randn(2, 8, 3, 3)

    monkeypatch.setattr(moe.dist, "is_available", lambda: True)
    monkeypatch.setattr(moe.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(moe.dist, "get_world_size", lambda: 1)

    def fail_all_reduce(*args, **kwargs):
        raise AssertionError("all_reduce should not run when world_size == 1")

    monkeypatch.setattr(moe.dist, "all_reduce", fail_all_reduce)

    router.train()
    router(x)


def test_semift_config_exposes_moe_fields():
    semift_text = (TUNERS_DIR / "semift.py").read_text()
    assert "moe_num_experts" in semift_text
    assert "moe_topk" in semift_text
    assert "moe_router_balance_mode" in semift_text
    assert "moe_router_bias_update_speed" in semift_text
    assert "moe_router_bias_clip" in semift_text
    assert "moe_num_prefix_tokens" in semift_text
    assert "moe_use_shared_expert" in semift_text
    assert "moe_conv_hidden_ratio" in semift_text
    assert "moe_conv_kernel_size" in semift_text
    assert "moe_conv_context_kernel_size" in semift_text
    assert "moe_conv_use_grn" in semift_text
    assert "moe_expert_scales" in semift_text
    assert "moe_conv_gate_temperature" in semift_text
    assert "moe_layerscale_init" in semift_text
    assert "moe_expert_drop_path_rate" in semift_text


def test_scale_gated_conv_expert_forward_with_explicit_hw():
    expert = moe.ScaleGatedConvExpert(
        r=8, hidden_ratio=2.0, kernel_size=3, context_kernel_size=5, scale=4
    )
    x = torch.randn(2, 15, 8)
    y = expert(x, hw=(3, 5))
    assert y.shape == x.shape


def test_scale_gated_conv_expert_gate_weights_are_normalized():
    expert = moe.ScaleGatedConvExpert(r=8, hidden_ratio=2.0, scale=2)
    x = torch.randn(1, 9, 8)
    bsz, n_tokens, _ = x.shape
    h, w = 3, 3
    x_norm = expert.pre_norm(x)
    value, gate = expert.pw_expand(x_norm).chunk(2, dim=-1)
    x_hidden = value * torch.nn.functional.silu(gate)
    x_2d = x_hidden.transpose(1, 2).reshape(bsz, expert.hidden_dim, h, w).contiguous()
    gate_logits = expert.branch_gate(x_2d).view(bsz, 2, expert.hidden_dim, h, w)
    gate_weight = torch.softmax(gate_logits, dim=1)
    assert torch.allclose(gate_weight.sum(dim=1), torch.ones_like(gate_weight[:, 0]), atol=1e-6)


def test_semift_scalegate_forward_shape_and_scales():
    module = moe.SemiFtScaleGate(
        in_features=16,
        out_features=16,
        r=8,
        num_experts=4,
        topk=2,
        num_prefix_tokens=5,
        scales=[1, 2, 4, 8],
    )
    module.train()

    x = torch.randn(2, 14, 16, requires_grad=True)
    y = module(x)

    assert y.shape == x.shape
    assert [expert.scale for expert in module.experts] == [1, 2, 4, 8]
    assert module.shared_expert.scale == 1

    loss = y.sum()
    loss.backward()

    assert module.experts[0].branch_gate.weight.grad is not None


def test_semift_scalegate_requires_scale_per_expert():
    try:
        moe.SemiFtScaleGate(
            in_features=16,
            out_features=16,
            r=8,
            num_experts=4,
            scales=[1, 2],
        )
    except ValueError as exc:
        assert "len(scales) == num_experts" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched scales")


def test_semift_samoe_forward_shape_and_sparse_selection():
    module = moe.SemiFtSAMoE(
        in_features=16,
        out_features=16,
        r=8,
        num_experts=4,
        topk=2,
        num_prefix_tokens=5,
        scales=[1, 2, 4, 8],
    )
    module.train()

    x = torch.randn(2, 14, 16, requires_grad=True)
    y = module(x)

    assert y.shape == x.shape
    assert module.router_logits.shape == (2, 4)
    assert module.selected_experts.shape == (2, 2)
    assert module.expert_load.shape == (4,)
    assert torch.allclose(module.expert_load.sum(), torch.tensor(1.0), atol=1e-6)

    loss = y.sum()
    loss.backward()

    assert module.proj_down.weight.grad is not None
    assert module.gating_network.router_mlp[2].weight.grad is not None
    assert module.proj_up.weight.grad is not None


def test_semift_samoe_sparse_dispatch_skips_unselected_experts():
    module = moe.SemiFtSAMoE(
        in_features=16,
        out_features=16,
        r=8,
        num_experts=4,
        topk=1,
        num_prefix_tokens=5,
        scales=[1, 2, 4, 8],
    )
    calls = [0, 0, 0, 0]
    original_forward = module.experts[0].forward

    def make_forward(idx):
        def wrapped(x_2d):
            calls[idx] += 1
            return original_forward(x_2d) if idx == 0 else module.experts[idx].__class__.forward(module.experts[idx], x_2d)
        return wrapped

    for idx, expert in enumerate(module.experts):
        expert.forward = make_forward(idx)

    with torch.no_grad():
        module.gating_network.router_mlp[2].weight.zero_()
        module.gating_network.router_mlp[2].weight[0, 0] = 10.0

    x = torch.randn(2, 14, 16)
    module(x)
    assert sum(calls) >= 1
    assert any(call == 0 for call in calls)


def test_channel_scale2d_scales_channels_not_spatial():
    scale = moe.ChannelScale2d(4, init_values=2.0)
    x = torch.ones(1, 4, 4, 4)
    y = scale(x)
    assert torch.allclose(y, torch.full_like(x, 2.0))
    assert y[0, 0, 0, 0].item() == y[0, 0, 3, 3].item()


def test_channel_scale2d_loads_legacy_layerscale_gamma_shape():
    scale = moe.ChannelScale2d(4, init_values=0.0)
    state_dict = {"gamma": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    scale.load_state_dict(state_dict, strict=True)
    assert tuple(scale.gamma.shape) == (1, 4, 1, 1)
    assert torch.allclose(scale.gamma.view(-1), torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_semift_samoe_eval_routing_ignores_expert_bias():
    module = moe.SemiFtSAMoE(
        in_features=16,
        out_features=16,
        r=8,
        num_experts=4,
        topk=1,
        num_prefix_tokens=5,
        scales=[1, 2, 4, 8],
    )
    module.eval()
    with torch.no_grad():
        module.gating_network.expert_bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
        module.gating_network.router_mlp[0].weight.zero_()
        module.gating_network.router_mlp[0].bias.zero_()
        module.gating_network.router_mlp[0].bias[0] = 1.0
        module.gating_network.router_mlp[2].weight.zero_()
        module.gating_network.router_mlp[2].weight[1, 0] = 5.0
    x = torch.randn(2, 14, 16)
    module(x)
    assert torch.all(module.selected_experts == 1)


def test_semift_samoe_layernorm2d_is_constructed():
    module = moe.SemiFtSAMoE(
        in_features=16,
        out_features=16,
        r=8,
        num_experts=4,
        topk=2,
        num_prefix_tokens=5,
        scales=[1, 2, 4, 8],
        conv_norm_type="layernorm",
    )
    assert isinstance(module.experts[0].norm, moe.LayerNorm2d)
    assert isinstance(module.shared_expert.norm, moe.LayerNorm2d)
