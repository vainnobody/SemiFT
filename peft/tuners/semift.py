# coding=utf-8
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import PeftConfig, PeftType
from .moe import SemiFt, SemiFtSAMoE, SemiFtSAMoEV4, SemiFtSAMoEV5, SemiFtScaleGate


@dataclass
class SemiFTConfig(PeftConfig):
    method: str = field(default="lora")
    target_modules: Optional[Union[List[str], str]] = field(default=None)
    bias: str = field(default="none")
    modules_to_save: Optional[List[str]] = field(default=None)

    # LoRA-family defaults
    r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)

    # SSF
    ssf_init_scale: float = field(default=1.0)
    ssf_init_shift_std: float = field(default=0.02)

    # AdaptFormer
    adapter_dim: int = field(default=64)
    adapter_dropout: float = field(default=0.1)
    adapter_scale: float = field(default=0.1)
    adapter_layernorm_option: str = field(default="none")

    # FacT
    fact_rank: int = field(default=8)
    fact_scale: float = field(default=1.0)
    fact_dropout: float = field(default=0.1)

    # Conv-LoRA / HydraLoRA
    conv_lora_kernel_size: int = field(default=3)
    conv_lora_dropout: float = field(default=0.1)
    hydra_num_branches: int = field(default=4)
    hydra_router_hidden: int = field(default=64)
    hydra_router_dropout: float = field(default=0.1)

    # SemiFT / MoE
    nclass: int = field(default=5)
    moe_num_experts: int = field(default=4)
    moe_topk: int = field(default=2)
    moe_router_balance_mode: str = field(default="deepseek_v3")
    moe_router_bias_update_speed: float = field(default=1e-3)
    moe_router_bias_clip: float = field(default=0.05)
    moe_router_aux_loss_coef: float = field(default=1e-2)
    moe_router_z_loss_coef: float = field(default=1e-3)
    moe_router_jitter_noise: float = field(default=1e-2)
    moe_num_prefix_tokens: int = field(default=-1)
    moe_use_shared_expert: bool = field(default=True)
    moe_conv_hidden_ratio: float = field(default=2.0)
    moe_conv_kernel_size: int = field(default=3)
    moe_conv_context_kernel_size: int = field(default=5)
    moe_conv_use_grn: bool = field(default=True)
    moe_conv_norm_type: str = field(default="layernorm")
    moe_expert_scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    moe_conv_gate_temperature: float = field(default=1.0)
    moe_layerscale_init: float = field(default=1e-5)
    moe_expert_drop_path_rate: float = field(default=0.0)
    moe_branch_gate_init_bias: float = field(default=-2.0)

    def __post_init__(self):
        self.peft_type = PeftType.LORA


METHOD_DEFAULT_TARGETS = {
    "semift": ["mlp"],
    "semift_samoe": ["mlp"],
    "samoev4": ["mlp"],
    "samoev5": ["mlp"],
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

HIGH_LEVEL_TO_SUBMODULES = {
    "attn": ["qkv", "proj"],
    "mlp": ["fc1", "fc2"],
}
BLOCK_LEVEL_METHODS = {"semift", "semift_samoe", "samoev4", "samoev5", "semift_scalegate", "adaptformer", "fact_tt", "fact_tk"}
PARAMETER_ONLY_METHODS = {"bitfit"}
SSF_METHODS = {"ssf"}


class AdaptModel(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._fact_tt_shared: Dict[Tuple[int, int], FactTTShared] = {}
        self._fact_tk_shared: Dict[Tuple[int, int], FactTKShared] = {}
        self._method_counters: Dict[Tuple[str, int, int], int] = {}
        self._find_and_replace()
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_4bit or loaded_in_8bit:
            raise ImportError(
                "To use PEFT adapters with 8-bit or 4-bit quantization, please install `bitsandbytes`."
            )

        matched = False
        expanded_targets = self._expand_target_modules(self.peft_config.target_modules)
        key_list = [key for key, _ in self.model.named_modules() if key]
        visited = set()

        for key in key_list:
            if key in visited:
                continue
            if not self._matches(key, expanded_targets):
                continue
            parent, target, target_name = self._get_submodules(key)
            if self.peft_config.method in PARAMETER_ONLY_METHODS:
                self._enable_bitfit(target_name, target)
                matched = True
                visited.add(key)
                continue

            if self.peft_config.method in BLOCK_LEVEL_METHODS and target_name in {"attn", "mlp"}:
                new_module = self._build_block_level_adapter(target_name, target)
                self._insert_module(parent, target_name, new_module)
                matched = True
                visited.add(key)
                continue

            if self.peft_config.method in SSF_METHODS:
                new_module = self._build_ssf_wrapper(target_name, target)
                self._insert_module(parent, target_name, new_module)
                matched = True
                visited.add(key)
                continue

            if target_name in HIGH_LEVEL_TO_SUBMODULES:
                for child_name in HIGH_LEVEL_TO_SUBMODULES[target_name]:
                    child_key = f"{key}.{child_name}"
                    child_parent = target
                    child_target = getattr(target, child_name, None)
                    if child_target is None:
                        continue
                    if not self._supports_leaf_adapter_target(child_name, child_target):
                        visited.add(child_key)
                        continue
                    new_module = self._build_leaf_adapter(child_name, child_target)
                    self._insert_module(child_parent, child_name, new_module)
                    visited.add(child_key)
                    matched = True
                visited.add(key)
                continue

            if not self._supports_leaf_adapter_target(target_name, target):
                visited.add(key)
                continue

            new_module = self._build_leaf_adapter(target_name, target)
            self._insert_module(parent, target_name, new_module)
            matched = True
            visited.add(key)

        if not matched:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model for method {self.peft_config.method}."
            )

    def _expand_target_modules(self, targets):
        if isinstance(targets, str):
            return targets
        if targets is None:
            targets = METHOD_DEFAULT_TARGETS.get(self.peft_config.method, ["mlp"])
        expanded = []
        for target in targets:
            target = str(target)
            if self.peft_config.method not in BLOCK_LEVEL_METHODS and target in HIGH_LEVEL_TO_SUBMODULES:
                expanded.extend(HIGH_LEVEL_TO_SUBMODULES[target])
                expanded.append(target)
            else:
                expanded.append(target)
        # preserve order, remove duplicates
        result = []
        for item in expanded:
            if item not in result:
                result.append(item)
        return result

    def _matches(self, key, targets):
        if isinstance(targets, str):
            return re.fullmatch(targets, key) is not None
        return any(key.endswith(target_key) for target_key in targets)

    def _build_block_level_adapter(self, target_name, target):
        input_dim, output_dim = self._infer_block_dims(target_name, target)
        if self.peft_config.method == "semift":
            adapter = SemiFt(input_dim, output_dim, **self._semift_kwargs())
        elif self.peft_config.method == "semift_samoe":
            adapter = SemiFtSAMoE(input_dim, output_dim, **self._semift_kwargs())
        elif self.peft_config.method == "samoev4":
            adapter = SemiFtSAMoEV4(input_dim, output_dim, **self._semift_kwargs())
        elif self.peft_config.method == "samoev5":
            adapter = SemiFtSAMoEV5(input_dim, output_dim, **self._semift_kwargs())
        elif self.peft_config.method == "semift_scalegate":
            adapter = SemiFtScaleGate(input_dim, output_dim, **self._semift_kwargs())
        elif self.peft_config.method == "adaptformer":
            adapter = AdapterFormer(
                input_dim,
                output_dim,
                r=self.peft_config.adapter_dim,
                dropout=self.peft_config.adapter_dropout,
                scale=self.peft_config.adapter_scale,
                layernorm_option=self.peft_config.adapter_layernorm_option,
            )
        elif self.peft_config.method == "fact_tt":
            shared = self._get_fact_tt_shared(input_dim, output_dim)
            adapter = FactTTAdapter(
                shared,
                dropout=self.peft_config.fact_dropout,
                scale=self.peft_config.fact_scale,
            )
        elif self.peft_config.method == "fact_tk":
            shared = self._get_fact_tk_shared(input_dim, output_dim)
            slice_index = self._next_method_counter((self.peft_config.method, input_dim, output_dim))
            adapter = FactTKAdapter(
                shared,
                slice_index=slice_index,
                dropout=self.peft_config.fact_dropout,
                scale=self.peft_config.fact_scale,
            )
        else:
            raise ValueError(f"Unsupported block-level method: {self.peft_config.method}")
        return WarpBlock(target, adapter)

    def _build_ssf_wrapper(self, target_name, target):
        output_dim = self._infer_output_dim(target_name, target)
        return SsfWrapper(
            target,
            output_dim=output_dim,
            init_scale=self.peft_config.ssf_init_scale,
            init_shift_std=self.peft_config.ssf_init_shift_std,
        )

    def _supports_leaf_adapter_target(self, target_name, target):
        if self.peft_config.method == "ssf":
            return True
        try:
            self._infer_linear_dims(target_name, target)
        except ValueError:
            return False
        return True

    def _build_leaf_adapter(self, target_name, target):
        input_dim, output_dim = self._infer_linear_dims(target_name, target)
        method = self.peft_config.method
        if method == "lora":
            adapter = Lora(
                input_dim,
                output_dim,
                r=self.peft_config.r,
                lora_alpha=self.peft_config.lora_alpha,
                p=self.peft_config.lora_dropout,
            )
            return WarpBlock(target, adapter)
        if method == "conv_lora":
            adapter = ConvLora(
                input_dim,
                output_dim,
                r=self.peft_config.r,
                lora_alpha=self.peft_config.lora_alpha,
                dropout=self.peft_config.conv_lora_dropout,
                kernel_size=self.peft_config.conv_lora_kernel_size,
                num_prefix_tokens=self.peft_config.moe_num_prefix_tokens,
            )
            return WarpBlock(target, adapter)
        if method == "hydralora":
            adapter = HydraLora(
                input_dim,
                output_dim,
                r=self.peft_config.r,
                num_branches=self.peft_config.hydra_num_branches,
                router_hidden=self.peft_config.hydra_router_hidden,
                router_dropout=self.peft_config.hydra_router_dropout,
                lora_alpha=self.peft_config.lora_alpha,
                dropout=self.peft_config.lora_dropout,
            )
            return WarpBlock(target, adapter)
        if method == "ssf":
            return self._build_ssf_wrapper(target_name, target)
        raise ValueError(f"Unsupported leaf adapter method: {method}")

    def _enable_bitfit(self, target_name, target):
        for param in target.parameters():
            param.requires_grad = False
        for name, param in target.named_parameters(recurse=True):
            if name.endswith("bias") or ".bias" in name:
                param.requires_grad = True

    def _semift_kwargs(self):
        num_prefix_tokens = self.peft_config.moe_num_prefix_tokens
        if num_prefix_tokens is None or int(num_prefix_tokens) <= 0:
            num_prefix_tokens = AdaptModel._infer_num_prefix_tokens_from_model(self)
        if self.peft_config.method in {"semift_samoe", "samoev4", "samoev5"}:
            kwargs = {
                "r": self.peft_config.r,
                "num_experts": self.peft_config.moe_num_experts,
                "topk": self.peft_config.moe_topk,
                "router_balance_mode": self.peft_config.moe_router_balance_mode,
                "router_bias_update_speed": self.peft_config.moe_router_bias_update_speed,
                "router_bias_clip": self.peft_config.moe_router_bias_clip,
                "router_jitter_noise": self.peft_config.moe_router_jitter_noise,
                "num_prefix_tokens": num_prefix_tokens,
                "use_shared_expert": self.peft_config.moe_use_shared_expert,
                "conv_kernel_size": self.peft_config.moe_conv_kernel_size,
                "conv_norm_type": self.peft_config.moe_conv_norm_type,
                "scales": self.peft_config.moe_expert_scales,
                "layerscale_init": self.peft_config.moe_layerscale_init,
                "drop_path_rate": self.peft_config.moe_expert_drop_path_rate,
            }
            if self.peft_config.method == "samoev5":
                kwargs["branch_gate_init_bias"] = self.peft_config.moe_branch_gate_init_bias
            return kwargs
        kwargs = {
            "r": self.peft_config.r,
            "num_experts": self.peft_config.moe_num_experts,
            "topk": self.peft_config.moe_topk,
            "router_balance_mode": self.peft_config.moe_router_balance_mode,
            "router_bias_update_speed": self.peft_config.moe_router_bias_update_speed,
            "router_bias_clip": self.peft_config.moe_router_bias_clip,
            "router_jitter_noise": self.peft_config.moe_router_jitter_noise,
            "num_prefix_tokens": num_prefix_tokens,
            "use_shared_expert": self.peft_config.moe_use_shared_expert,
            "conv_hidden_ratio": self.peft_config.moe_conv_hidden_ratio,
            "conv_kernel_size": self.peft_config.moe_conv_kernel_size,
            "conv_context_kernel_size": self.peft_config.moe_conv_context_kernel_size,
            "conv_use_grn": self.peft_config.moe_conv_use_grn,
            "conv_norm_type": self.peft_config.moe_conv_norm_type,
            "scales": self.peft_config.moe_expert_scales,
            "conv_gate_temperature": self.peft_config.moe_conv_gate_temperature,
        }
        return kwargs

    def _infer_num_prefix_tokens_from_model(self):
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            return 1
        if hasattr(backbone, "num_register_tokens"):
            return 1 + int(getattr(backbone, "num_register_tokens"))
        if hasattr(backbone, "n_storage_tokens"):
            return 1 + int(getattr(backbone, "n_storage_tokens"))
        return 1

    def _infer_block_dims(self, target_name, target):
        if target_name == "attn":
            return target.qkv.in_features, target.proj.out_features
        if target_name == "mlp":
            return target.fc1.in_features, target.fc2.out_features
        return self._infer_linear_dims(target_name, target)

    def _infer_linear_dims(self, target_name, target):
        if isinstance(target, nn.Linear):
            return target.in_features, target.out_features
        if target_name == "patch_embed" and hasattr(target, "proj"):
            proj = target.proj
            in_features = proj.in_channels * proj.kernel_size[0] * proj.kernel_size[1]
            return in_features, proj.out_channels
        if hasattr(target, "weight") and target.weight.ndim == 2:
            return target.weight.shape[1], target.weight.shape[0]
        raise ValueError(f"Cannot infer input/output dimensions for target '{target_name}' ({type(target)}).")

    def _infer_output_dim(self, target_name, target):
        if target_name == "attn":
            return target.proj.out_features
        if target_name == "mlp":
            return target.fc2.out_features
        if hasattr(target, "normalized_shape"):
            shape = target.normalized_shape
            return int(shape[0] if isinstance(shape, (list, tuple)) else shape)
        if isinstance(target, nn.Linear):
            return target.out_features
        if target_name == "patch_embed" and hasattr(target, "proj"):
            return target.proj.out_channels
        if hasattr(target, "weight"):
            if target.weight.ndim >= 1:
                return int(target.weight.shape[0])
        raise ValueError(f"Cannot infer output dimension for target '{target_name}' ({type(target)}).")

    def _get_fact_tt_shared(self, input_dim, output_dim):
        key = (input_dim, output_dim)
        if key not in self._fact_tt_shared:
            self._fact_tt_shared[key] = FactTTShared(
                input_dim,
                output_dim,
                rank=self.peft_config.fact_rank,
            )
            self.add_module(f"fact_tt_shared_{input_dim}_{output_dim}", self._fact_tt_shared[key])
        return self._fact_tt_shared[key]

    def _get_fact_tk_shared(self, input_dim, output_dim):
        key = (input_dim, output_dim)
        if key not in self._fact_tk_shared:
            self._fact_tk_shared[key] = FactTKShared(
                input_dim,
                output_dim,
                rank=self.peft_config.fact_rank,
            )
            self.add_module(f"fact_tk_shared_{input_dim}_{output_dim}", self._fact_tk_shared[key])
        return self._fact_tk_shared[key]

    def _next_method_counter(self, key):
        idx = self._method_counters.get(key, 0)
        self._method_counters[key] = idx + 1
        return idx

    def _insert_module(self, parent_module, child_name, new_module):
        setattr(parent_module, child_name, new_module)

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in asdict(self.peft_config).items()
        }
        if inference:
            config["inference_mode"] = True
        return config


class LoraLayer:
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else lambda x: x
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Lora(nn.Module):
    def __init__(self, in_features, out_features, r=32, lora_alpha=64, p=0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(p)
        self.scaling = self.lora_alpha / max(self.r, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class WarpBlock(nn.Module):
    def __init__(self, base_layer, adapter):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x, *args, **kwargs):
        return self.base_layer(x, *args, **kwargs) + self.adapter(x)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)


class AdapterFormer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        dropout: float = 0.1,
        scale: float = 0.1,
        layernorm_option: str = "none",
    ):
        super().__init__()
        self.layernorm_option = layernorm_option
        self.adapter_layer_norm_before = None
        if layernorm_option in {"in", "out"}:
            self.adapter_layer_norm_before = nn.LayerNorm(in_features)
        self.down_proj = nn.Linear(in_features, r, bias=True)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(r, out_features, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        if self.layernorm_option == "in" and self.adapter_layer_norm_before is not None:
            x = self.adapter_layer_norm_before(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        if self.layernorm_option == "out" and self.adapter_layer_norm_before is not None:
            x = self.adapter_layer_norm_before(x)
        return x * self.scale


class SsfWrapper(nn.Module):
    def __init__(self, base_layer, output_dim: int, init_scale: float = 1.0, init_shift_std: float = 0.02):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.scale = nn.Parameter(torch.ones(output_dim) * init_scale)
        self.shift = nn.Parameter(torch.zeros(output_dim))
        if init_shift_std > 0:
            nn.init.normal_(self.shift, std=init_shift_std)
            nn.init.normal_(self.scale, mean=init_scale, std=min(init_shift_std, 0.02))

    def forward(self, x, *args, **kwargs):
        out = self.base_layer(x, *args, **kwargs)
        if out.dim() >= 3 and out.shape[-1] == self.scale.shape[0]:
            return out * self.scale + self.shift
        if out.dim() == 4 and out.shape[1] == self.scale.shape[0]:
            return out * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        raise ValueError("SSF wrapper got unsupported output shape.")


class FactTTShared(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int):
        super().__init__()
        self.fac_tu = nn.Linear(input_dim, rank, bias=False)
        self.fac_tv = nn.Linear(rank, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.fac_tu.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fac_tv.weight)


class FactTTAdapter(nn.Module):
    def __init__(self, shared: FactTTShared, dropout: float = 0.1, scale: float = 1.0):
        super().__init__()
        self.shared = shared
        rank = shared.fac_tu.out_features
        self.middle = nn.Linear(rank, rank, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        nn.init.eye_(self.middle.weight)

    def forward(self, x):
        return self.shared.fac_tv(self.middle(self.dropout(self.shared.fac_tu(x)))) * self.scale


class FactTKShared(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int):
        super().__init__()
        self.rank = rank
        self.fac_tu = nn.Linear(input_dim, rank, bias=False)
        self.fac_tv = nn.Linear(rank, output_dim, bias=False)
        self.tensor_core = nn.Parameter(torch.empty(rank, rank, rank))
        self.selector_bank = nn.Parameter(torch.empty(rank, max(rank * 4, 16)))
        nn.init.kaiming_uniform_(self.fac_tu.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fac_tv.weight)
        nn.init.xavier_uniform_(self.tensor_core)
        nn.init.xavier_uniform_(self.selector_bank)

    def get_matrix(self, slice_index: int):
        selector = self.selector_bank[:, slice_index % self.selector_bank.shape[1]]
        return torch.einsum("abc,c->ab", self.tensor_core, selector)


class FactTKAdapter(nn.Module):
    def __init__(self, shared: FactTKShared, slice_index: int, dropout: float = 0.1, scale: float = 1.0):
        super().__init__()
        self.shared = shared
        self.slice_index = slice_index
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def forward(self, x):
        hidden = self.shared.fac_tu(x)
        matrix = self.shared.get_matrix(self.slice_index)
        hidden = hidden @ matrix
        hidden = self.dropout(hidden)
        return self.shared.fac_tv(hidden) * self.scale


class ConvLora(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=32, dropout=0.1, kernel_size=3, num_prefix_tokens=5):
        super().__init__()
        self.lora = Lora(in_features, out_features, r=r, lora_alpha=lora_alpha, p=dropout)
        self.num_prefix_tokens = num_prefix_tokens
        self.spatial_proj = nn.Linear(in_features, out_features, bias=False)
        self.depthwise = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_features,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.spatial_proj.weight)

    def forward(self, x):
        delta = self.lora(x)
        if x.dim() != 3:
            return delta
        bsz, tokens, channels = x.shape
        spatial_tokens = tokens - self.num_prefix_tokens
        side = int(math.sqrt(max(spatial_tokens, 0)))
        if spatial_tokens <= 0 or side * side != spatial_tokens:
            return delta
        prefix = delta[:, : self.num_prefix_tokens]
        spatial_x = x[:, self.num_prefix_tokens :].transpose(1, 2).reshape(bsz, channels, side, side)
        conv_out = self.depthwise(spatial_x).flatten(2).transpose(1, 2)
        conv_out = self.spatial_proj(self.dropout(conv_out))
        return torch.cat([prefix, delta[:, self.num_prefix_tokens :] + conv_out], dim=1)


class HydraLora(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        r=8,
        num_branches=4,
        router_hidden=64,
        router_dropout=0.1,
        lora_alpha=32,
        dropout=0.1,
    ):
        super().__init__()
        self.shared_A = nn.Linear(in_features, r, bias=False)
        self.branches = nn.ModuleList([nn.Linear(r, out_features, bias=False) for _ in range(num_branches)])
        self.router = nn.Sequential(
            nn.Linear(in_features, router_hidden),
            nn.GELU(),
            nn.Dropout(router_dropout),
            nn.Linear(router_hidden, num_branches),
        )
        self.dropout = nn.Dropout(dropout)
        self.scaling = lora_alpha / max(r, 1)
        nn.init.kaiming_uniform_(self.shared_A.weight, a=math.sqrt(5))
        for branch in self.branches:
            nn.init.zeros_(branch.weight)

    def forward(self, x):
        hidden = self.shared_A(self.dropout(x))
        logits = self.router(x)
        weights = torch.softmax(logits, dim=-1)
        outputs = torch.stack([branch(hidden) for branch in self.branches], dim=-1)
        return (outputs * weights.unsqueeze(-2)).sum(dim=-1) * self.scaling
