# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose
from .moe import SemiFt


@dataclass
class SemiFTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    method: str = field(default="lora")

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    nclass: int = field(default=5, metadata={"help": "Numbers of classes"})
    moe_num_experts: int = field(default=4, metadata={"help": "Number of routed experts in SemiFT MoE adapter"})
    moe_topk: int = field(default=2, metadata={"help": "Top-k experts selected per token in SemiFT MoE adapter"})
    moe_router_aux_loss_coef: float = field(default=1e-2, metadata={"help": "Load balancing auxiliary loss coefficient for SemiFT router"})
    moe_router_z_loss_coef: float = field(default=1e-3, metadata={"help": "Router z-loss coefficient for SemiFT router"})
    moe_router_jitter_noise: float = field(default=1e-2, metadata={"help": "Input jitter noise for SemiFT router during training"})
    moe_num_prefix_tokens: int = field(default=5, metadata={"help": "Number of prefix tokens excluded from sparse routing"})
    moe_use_shared_expert: bool = field(default=True, metadata={"help": "Whether SemiFT adapter uses a shared dense expert branch"})
    moe_conv_hidden_ratio: float = field(default=2.0, metadata={"help": "Expansion ratio used inside ConvExpert"})
    moe_conv_kernel_size: int = field(default=3, metadata={"help": "Local depthwise kernel size used inside ConvExpert"})
    moe_conv_context_kernel_size: int = field(default=5, metadata={"help": "Context depthwise kernel size used inside ConvExpert"})
    moe_conv_use_grn: bool = field(default=True, metadata={"help": "Whether ConvExpert uses GRN normalization"})
    moe_conv_norm_type: str = field(default="layernorm", metadata={"help": "Normalization type used after ConvExpert convolution mixing"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class AdaptModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):  # LoraConfig, CasualLM
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        # mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_4bit or loaded_in_8bit:
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        # kwargs = {
        #     "r": self.peft_config.r,
        #     "lora_alpha": self.peft_config.lora_alpha,
        #     "lora_dropout": self.peft_config.lora_dropout,
        #     "lora_nums": self.peft_config.lora_nums,
        #     "fan_in_fan_out": self.peft_config.fan_in_fan_out,
        #     "merge_weights": (
        #         self.peft_config.merge_weights or self.peft_config.inference_mode
        #     )
        #     and not is_hf_device_map_available,
        # }
        kwargs = {
            "r": 32,
            "num_experts": self.peft_config.moe_num_experts,
            "topk": self.peft_config.moe_topk,
            "router_aux_loss_coef": self.peft_config.moe_router_aux_loss_coef,
            "router_z_loss_coef": self.peft_config.moe_router_z_loss_coef,
            "router_jitter_noise": self.peft_config.moe_router_jitter_noise,
            "num_prefix_tokens": self.peft_config.moe_num_prefix_tokens,
            "use_shared_expert": self.peft_config.moe_use_shared_expert,
            "conv_hidden_ratio": self.peft_config.moe_conv_hidden_ratio,
            "conv_kernel_size": self.peft_config.moe_conv_kernel_size,
            "conv_context_kernel_size": self.peft_config.moe_conv_context_kernel_size,
            "conv_use_grn": self.peft_config.moe_conv_use_grn,
            "conv_norm_type": self.peft_config.moe_conv_norm_type,
        }

        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(
                    key.endswith(target_key)
                    for target_key in self.peft_config.target_modules
                )
            if target_module_found:  # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)

                if self.peft_config.method == "semift":
                    if target_name == "attn":
                        new_module = SemiFt(
                            target.qkv.in_features, target.qkv.out_features, **kwargs
                        )
                    elif target_name == "mlp":
                        new_module = SemiFt(
                            target.fc1.in_features, target.fc2.out_features, **kwargs
                        )
                elif self.peft_config.method == "lora":
                    new_module = Lora(
                        target.in_features, target.out_features, r=32, lora_alpha=64
                    )

                new_module = WarpBlock(target, new_module)
                self._insert_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _insert_module(self, parent_module, child_name, new_module, old_module):
        old_module = new_module
        setattr(parent_module, child_name, old_module)

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
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

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.lora_num = lora_nums

        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            setattr(self, f"lora_A", nn.Linear(in_features, r, bias=False))
            for i in range(self.lora_num):
                setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(getattr(self, f"lora_A").weight, a=math.sqrt(5))
            for i in range(self.lora_num):
                nn.init.zeros_(getattr(self, f"lora_B{i}").weight)

            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_route.train(mode)
        getattr(self, f"lora_A").train(mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lora_route.eval()
        getattr(self, f"lora_A").eval()
        for i in range(self.lora_num):
            getattr(self, f"lora_B{i}").eval()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x: torch.Tensor, task_types=None):

        if self.disable_adapters:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
            raise ImportError(":(")
        elif self.r > 0 and not self.merged:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

            if self.r > 0:
                route_weight = nn.functional.softmax(
                    self.lora_route(x), dim=-1, dtype=torch.float32
                ).to(result.dtype)

                for i in range(self.lora_num):
                    result = (
                        result
                        + torch.unsqueeze(route_weight[:, :, i], -1)
                        * getattr(self, f"lora_B{i}")(
                            getattr(self, f"lora_A")(self.lora_dropout(x))
                        )
                        * self.scaling
                    )

        blcls = torch.zeros(1)[0].to(result)

        return result, blcls


class Lora(nn.Module):
    def __init__(self, in_features, out_features, r=32, lora_alpha=64, p=0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        self.lora_dropout = nn.Dropout(p)
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return out


class WarpBlock(nn.Module):
    def __init__(self, base_layer, adapter):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base_layer(x) + self.adapter(x)


class AdapterFormer(nn.Module):
    """
    AdapterFormer: Optimized adapter module based on AdaptFormer paper (NeurIPS 2022)

    Key optimizations:
    1. Dropout placed after activation (not on input) - better regularization
    2. ReLU activation (consistent with original paper)
    3. Fixed scale factor (0.1) instead of learnable parameter - more stable training
    4. Added bias parameters - better expressiveness
    5. Improved initialization strategy - matches paper's best practices

    Paper: AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition
    ArXiv: https://arxiv.org/abs/2205.13535
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        dropout: float = 0.1,
        scale: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scale = scale  # Fixed scale value (not learnable)

        # Down projection: in_features -> r (with bias)
        self.down_proj = nn.Linear(in_features, r, bias=True)

        # Activation function: ReLU (consistent with AdaptFormer paper)
        self.act = nn.ReLU()

        # Up projection: r -> out_features (with bias)
        self.up_proj = nn.Linear(r, out_features, bias=True)

        # Dropout layer (will be applied after activation in forward)
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Improved initialization strategy based on AdaptFormer paper:
        - Kaiming uniform for down_proj (with gain sqrt(5))
        - Zero initialization for down_proj bias
        - Zero initialization for up_proj weights and bias
        - This ensures the adapter starts close to identity
        """
        # Down projection: Kaiming uniform (He initialization)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)

        # Up projection: Zero initialization
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimized dropout placement.

        Args:
            x: Input tensor of shape (B, N, in_features) or (N, in_features)

        Returns:
            output: Adapted tensor of shape (B, N, out_features) or (N, out_features)
        """
        # Down projection
        x = self.down_proj(x)

        # Activation
        x = self.act(x)

        # Dropout applied AFTER activation (optimized placement)
        x = self.dropout(x)

        # Up projection
        x = self.up_proj(x)

        # Scale with fixed factor
        x = x * self.scale

        return x
