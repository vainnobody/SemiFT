import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class GatedMLPExpert(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        return self.down_proj(x)


class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class GatingNetwork(nn.Module):
    def __init__(
        self,
        r,
        num_experts,
        topk=2,
        jitter_noise=0.01,
        balance_mode="deepseek_v3",
        bias_update_speed=1e-3,
        bias_clip=0.05,
    ):
        super().__init__()
        self.r = r
        self.num_experts = num_experts
        self.topk = max(1, min(topk, num_experts))
        self.jitter_noise = jitter_noise
        self.balance_mode = balance_mode
        self.bias_update_speed = bias_update_speed
        self.bias_clip = bias_clip

        self.norm = nn.LayerNorm(r)
        self.gate_proj = nn.Linear(r, num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer("expert_load", torch.zeros(num_experts, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))

    def _compute_expert_load(self, topk_idx):
        expert_mask = F.one_hot(topk_idx.reshape(-1), num_classes=self.num_experts).float()
        selected_counts = expert_mask.sum(dim=0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(selected_counts, op=dist.ReduceOp.SUM)
        total_selected = selected_counts.sum().clamp_min(1.0)
        return selected_counts / total_selected

    @torch.no_grad()
    def _update_expert_bias(self, expert_load):
        if self.balance_mode != "deepseek_v3" or self.num_experts <= 1:
            return
        target = expert_load.new_full((self.num_experts,), 1.0 / self.num_experts)
        direction = torch.sign(target - expert_load)
        updated_bias = self.expert_bias + direction * self.bias_update_speed
        updated_bias = updated_bias.clamp_(-self.bias_clip, self.bias_clip)
        self.expert_bias.copy_(updated_bias)
        self.expert_load.copy_(expert_load)

    def forward(self, x):
        bsz, n_tokens, dim = x.shape
        flat_x = x.reshape(-1, dim)
        flat_x = self.norm(flat_x)

        if self.training and self.jitter_noise > 0:
            noise = torch.empty_like(flat_x).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
            flat_x = flat_x * noise

        router_logits = self.gate_proj(flat_x)
        router_probs = torch.softmax(router_logits.float(), dim=-1).to(flat_x.dtype)
        selection_scores = router_probs.float() + self.expert_bias.unsqueeze(0)

        _, topk_idx = torch.topk(selection_scores, k=self.topk, dim=-1, sorted=False)
        topk_weight = torch.gather(router_probs, dim=-1, index=topk_idx)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(topk_weight.dtype).eps)

        aux_loss = router_logits.new_zeros(())
        z_loss = router_logits.new_zeros(())
        expert_load = self._compute_expert_load(topk_idx)
        self.expert_load.copy_(expert_load)
        if self.training and self.num_experts > 1:
            self._update_expert_bias(expert_load)

        stats = {
            'aux_loss': aux_loss.to(x.dtype),
            'z_loss': z_loss.to(x.dtype),
            'router_logits': router_logits.view(bsz, n_tokens, self.num_experts),
            'router_probs': router_probs.view(bsz, n_tokens, self.num_experts),
            'selection_scores': selection_scores.view(bsz, n_tokens, self.num_experts),
            'expert_bias': self.expert_bias.detach().clone(),
            'expert_load': self.expert_load.detach().clone(),
        }
        return (
            topk_idx.view(bsz, n_tokens, self.topk),
            topk_weight.view(bsz, n_tokens, self.topk),
            stats,
        )


class ConvExpert(nn.Module):
    def __init__(
        self,
        r,
        hidden_ratio=2.0,
        kernel_size=3,
        context_kernel_size=5,
        dropout=0.0,
        use_grn=True,
        norm_type="layernorm",
    ):
        super().__init__()
        self.r = r
        self.hidden_dim = max(int(r * hidden_ratio), r)
        self.kernel_size = kernel_size
        self.context_kernel_size = context_kernel_size
        self.use_grn = use_grn
        self.norm_type = norm_type

        self.pre_norm = nn.LayerNorm(r)
        self.pw_expand = nn.Linear(r, self.hidden_dim * 2, bias=False)
        self.dwconv_local = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=self.hidden_dim,
            bias=True,
        )
        self.dwconv_context = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=context_kernel_size,
            stride=1,
            padding=context_kernel_size // 2,
            groups=self.hidden_dim,
            bias=True,
        )
        self.mid_norm = nn.LayerNorm(self.hidden_dim) if norm_type == "layernorm" else nn.Identity()
        self.grn = GRN(self.hidden_dim) if use_grn else nn.Identity()
        self.pw_project = nn.Linear(self.hidden_dim, r, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.pw_expand.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dwconv_local.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dwconv_context.weight, a=math.sqrt(5))
        nn.init.zeros_(self.dwconv_local.bias)
        nn.init.zeros_(self.dwconv_context.bias)
        nn.init.zeros_(self.pw_project.weight)

    def _resolve_hw(self, n_tokens, hw=None):
        if hw is not None:
            if isinstance(hw, torch.Size):
                hw = tuple(hw)
            if isinstance(hw, (list, tuple)) and len(hw) == 2:
                h, w = int(hw[0]), int(hw[1])
            else:
                raise ValueError(f"hw must be a tuple/list of length 2, got {hw}")
            if h * w != n_tokens:
                raise ValueError(f"Invalid hw={hw} for {n_tokens} patch tokens")
            return h, w

        h = int(math.sqrt(n_tokens))
        w = h
        if h * w != n_tokens:
            raise ValueError(
                f"Cannot infer square patch grid from {n_tokens} tokens. "
                "Please pass hw=(H, W) to SemiFt.forward(..., hw=...)."
            )
        return h, w

    def forward(self, x, hw=None):
        if x.numel() == 0:
            return x
        bsz, n_tokens, channels = x.shape
        if channels != self.r:
            raise ValueError(f"Expected channel dim {self.r}, got {channels}")

        h, w = self._resolve_hw(n_tokens, hw)
        x_norm = self.pre_norm(x)
        value, gate = self.pw_expand(x_norm).chunk(2, dim=-1)
        x_hidden = value * F.silu(gate)

        x_2d = x_hidden.transpose(1, 2).reshape(bsz, self.hidden_dim, h, w).contiguous()
        x_local = self.dwconv_local(x_2d)
        x_context = self.dwconv_context(x_2d)
        x_2d = x_2d + x_local + x_context

        x_hidden = x_2d.permute(0, 2, 3, 1).contiguous()
        x_hidden = self.grn(x_hidden)
        x_hidden = self.mid_norm(x_hidden)
        x_hidden = self.pw_project(x_hidden.view(bsz, n_tokens, self.hidden_dim))
        x_hidden = self.dropout(x_hidden)
        return x_hidden


class SemiFt(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        r,
        num_experts=4,
        topk=2,
        num_prefix_tokens=5,
        router_jitter_noise=0.01,
        router_balance_mode="deepseek_v3",
        router_bias_update_speed=1e-3,
        router_bias_clip=0.05,
        use_shared_expert=True,
        expert_dropout=0.0,
        scales=None,
        conv_hidden_ratio=2.0,
        conv_kernel_size=3,
        conv_context_kernel_size=5,
        conv_use_grn=True,
        conv_norm_type="layernorm",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.num_experts = num_experts
        self.topk = max(1, min(topk, num_experts))
        self.num_prefix_tokens = num_prefix_tokens
        self.scales = scales or [1, 2, 4, 8]
        self.use_shared_expert = use_shared_expert

        self.proj_down = nn.Linear(in_features, r, bias=False)
        self.proj_up = nn.Linear(r, out_features, bias=False)
        self.input_act = nn.GELU()
        self.ls = LayerScale(out_features, init_values=1.0)

        expert_kwargs = dict(
            r=r,
            hidden_ratio=conv_hidden_ratio,
            kernel_size=conv_kernel_size,
            context_kernel_size=conv_context_kernel_size,
            dropout=expert_dropout,
            use_grn=conv_use_grn,
            norm_type=conv_norm_type,
        )
        self.shared_expert = ConvExpert(**expert_kwargs) if use_shared_expert else None
        self.experts = nn.ModuleList([ConvExpert(**expert_kwargs) for _ in range(num_experts)])

        self.gating_network = GatingNetwork(
            r=r,
            num_experts=num_experts,
            topk=self.topk,
            jitter_noise=router_jitter_noise,
            balance_mode=router_balance_mode,
            bias_update_speed=router_bias_update_speed,
            bias_clip=router_bias_clip,
        )
        self.aux_loss = None
        self.router_aux_loss = None
        self.router_z_loss = None
        self.router_logits = None
        self.router_probs = None
        self.selection_scores = None
        self.expert_bias = None
        self.expert_load = None
        self.last_hw = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.proj_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.proj_up.weight)

    def _resolve_hw(self, n_tokens, hw=None):
        if hw is not None:
            if isinstance(hw, torch.Size):
                hw = tuple(hw)
            if isinstance(hw, (list, tuple)) and len(hw) == 2:
                h, w = int(hw[0]), int(hw[1])
            else:
                raise ValueError(f"hw must be a tuple/list of length 2, got {hw}")
            if h * w != n_tokens:
                raise ValueError(f"Invalid hw={hw} for {n_tokens} patch tokens")
            return h, w

        h = int(math.sqrt(n_tokens))
        w = h
        if h * w != n_tokens:
            raise ValueError(
                f"Cannot infer square patch grid from {n_tokens} tokens in SemiFt. "
                "Please pass hw=(H, W) to the adapter forward call."
            )
        return h, w

    def _dense_moe_forward(self, patch_tokens, topk_idx, topk_weight, hw):
        router_weight = patch_tokens.new_zeros(
            patch_tokens.shape[0], patch_tokens.shape[1], self.num_experts
        )
        router_weight.scatter_add_(-1, topk_idx, topk_weight)

        sparse_out = patch_tokens.new_zeros(patch_tokens.shape)
        for expert_idx, expert in enumerate(self.experts):
            expert_out = expert(patch_tokens, hw=hw)
            sparse_out = sparse_out + expert_out * router_weight[..., expert_idx : expert_idx + 1]
        return sparse_out

    def forward(self, x, hw=None):
        x = self.input_act(self.proj_down(x))

        prefix = x[:, : self.num_prefix_tokens, :]
        patch_tokens = x[:, self.num_prefix_tokens :, :]

        if patch_tokens.numel() == 0:
            out = self.ls(self.proj_up(x))
            zero = x.new_zeros(())
            self.aux_loss = zero
            self.router_aux_loss = zero
            self.router_z_loss = zero
            self.router_logits = None
            self.router_probs = None
            self.selection_scores = None
            self.expert_bias = self.gating_network.expert_bias.detach().clone()
            self.expert_load = self.gating_network.expert_load.detach().clone()
            self.last_hw = None
            return out

        resolved_hw = self._resolve_hw(patch_tokens.shape[1], hw=hw)
        self.last_hw = resolved_hw
        residual = patch_tokens

        topk_idx, topk_weight, router_stats = self.gating_network(patch_tokens)
        sparse_out = self._dense_moe_forward(patch_tokens, topk_idx, topk_weight, hw=resolved_hw)

        shared_out = self.shared_expert(patch_tokens, hw=resolved_hw) if self.shared_expert is not None else 0.0
        combined = residual + shared_out + sparse_out

        zero = patch_tokens.new_zeros(())

        self.router_aux_loss = zero
        self.router_z_loss = zero
        self.router_logits = router_stats['router_logits']
        self.router_probs = router_stats['router_probs']
        self.selection_scores = router_stats['selection_scores']
        self.expert_bias = router_stats['expert_bias']
        self.expert_load = router_stats['expert_load']
        self.aux_loss = zero

        x = torch.cat([prefix, combined], dim=1)
        x = self.ls(self.proj_up(x))
        return x




class ScaleGatedConvExpert(ConvExpert):
    def __init__(
        self,
        r,
        hidden_ratio=2.0,
        kernel_size=3,
        context_kernel_size=5,
        dropout=0.0,
        use_grn=True,
        norm_type="layernorm",
        scale=1,
        gate_temperature=1.0,
    ):
        super().__init__(
            r=r,
            hidden_ratio=hidden_ratio,
            kernel_size=kernel_size,
            context_kernel_size=context_kernel_size,
            dropout=dropout,
            use_grn=use_grn,
            norm_type=norm_type,
        )
        self.scale = max(int(scale), 1)
        self.gate_temperature = float(gate_temperature)
        self.branch_gate = nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.branch_gate.weight)
        nn.init.zeros_(self.branch_gate.bias)

    def _scaled_context(self, x_2d):
        if self.scale <= 1:
            return self.dwconv_context(x_2d)

        h, w = x_2d.shape[-2:]
        pooled_h = max(1, h // self.scale)
        pooled_w = max(1, w // self.scale)
        pooled = F.adaptive_avg_pool2d(x_2d, output_size=(pooled_h, pooled_w))
        context = self.dwconv_context(pooled)
        if context.shape[-2:] != (h, w):
            context = F.interpolate(context, size=(h, w), mode="bilinear", align_corners=False)
        return context

    def forward(self, x, hw=None):
        if x.numel() == 0:
            return x
        bsz, n_tokens, channels = x.shape
        if channels != self.r:
            raise ValueError(f"Expected channel dim {self.r}, got {channels}")

        h, w = self._resolve_hw(n_tokens, hw)
        x_norm = self.pre_norm(x)
        value, gate = self.pw_expand(x_norm).chunk(2, dim=-1)
        x_hidden = value * F.silu(gate)

        x_2d = x_hidden.transpose(1, 2).reshape(bsz, self.hidden_dim, h, w).contiguous()
        x_local = self.dwconv_local(x_2d)
        x_context = self._scaled_context(x_2d)

        gate_logits = self.branch_gate(x_2d).view(bsz, 2, self.hidden_dim, h, w)
        gate_logits = gate_logits / max(self.gate_temperature, 1e-6)
        gate_weight = torch.softmax(gate_logits, dim=1)
        local_weight = gate_weight[:, 0]
        context_weight = gate_weight[:, 1]
        x_2d = x_2d + local_weight * x_local + context_weight * x_context

        x_hidden = x_2d.permute(0, 2, 3, 1).contiguous()
        x_hidden = self.grn(x_hidden)
        x_hidden = self.mid_norm(x_hidden)
        x_hidden = self.pw_project(x_hidden.view(bsz, n_tokens, self.hidden_dim))
        x_hidden = self.dropout(x_hidden)
        return x_hidden


class SemiFtScaleGate(SemiFt):
    def __init__(
        self,
        in_features,
        out_features,
        r,
        num_experts=4,
        topk=2,
        num_prefix_tokens=5,
        router_jitter_noise=0.01,
        router_balance_mode="deepseek_v3",
        router_bias_update_speed=1e-3,
        router_bias_clip=0.05,
        use_shared_expert=True,
        expert_dropout=0.0,
        scales=None,
        conv_hidden_ratio=2.0,
        conv_kernel_size=3,
        conv_context_kernel_size=5,
        conv_use_grn=True,
        conv_norm_type="layernorm",
        conv_gate_temperature=1.0,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            r=r,
            num_experts=num_experts,
            topk=topk,
            num_prefix_tokens=num_prefix_tokens,
            router_jitter_noise=router_jitter_noise,
            router_balance_mode=router_balance_mode,
            router_bias_update_speed=router_bias_update_speed,
            router_bias_clip=router_bias_clip,
            use_shared_expert=use_shared_expert,
            expert_dropout=expert_dropout,
            scales=scales,
            conv_hidden_ratio=conv_hidden_ratio,
            conv_kernel_size=conv_kernel_size,
            conv_context_kernel_size=conv_context_kernel_size,
            conv_use_grn=conv_use_grn,
            conv_norm_type=conv_norm_type,
        )
        if len(self.scales) != self.num_experts:
            raise ValueError(
                f"Expected len(scales) == num_experts, got {len(self.scales)} and {self.num_experts}"
            )
        expert_kwargs = dict(
            r=r,
            hidden_ratio=conv_hidden_ratio,
            kernel_size=conv_kernel_size,
            context_kernel_size=conv_context_kernel_size,
            dropout=expert_dropout,
            use_grn=conv_use_grn,
            norm_type=conv_norm_type,
            gate_temperature=conv_gate_temperature,
        )
        self.shared_expert = ScaleGatedConvExpert(scale=1, **expert_kwargs) if use_shared_expert else None
        self.experts = nn.ModuleList([
            ScaleGatedConvExpert(scale=scale, **expert_kwargs) for scale in self.scales
        ])

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 1. 深度卷积 (Depthwise Convolution)
        # 每个通道使用独立的卷积核，groups = in_channels
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=True,
        )

        # 2. 逐点卷积 (Pointwise Convolution)
        # 使用 1x1 卷积核来融合深度卷积后的通道信息
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.pointwise.weight, a=math.sqrt(5))
        nn.init.constant_(self.depthwise.bias, 0)
        nn.init.constant_(self.pointwise.bias, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LegacyConvExpert(nn.Module):
    def __init__(
        self,
        r,
        kernel_size=3,
        use_norm=True,
        activate="gelu",
        temperature=1.0,
        scale=2.0,
    ):
        super().__init__()
        self.r = r
        self.kernel_size = kernel_size
        self.use_norm = use_norm
        self.activate = activate
        self.temperature = temperature
        self.scale = scale

        if activate == "gelu":
            self.act = nn.GELU()
        elif activate == "relu":
            self.act = nn.ReLU()
        elif activate == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {activate}")

        self.conv1 = DWConv(
            r,
            r,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.conv2 = DWConv(
            r, r, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

        self.dropout = nn.Dropout(0.1)

        if use_norm:
            self.norm1 = nn.LayerNorm(r)
            self.norm2 = nn.LayerNorm(r)

        self._init_parameters()

    def _init_parameters(self):
        if self.use_norm and hasattr(self, "norm1"):
            nn.init.ones_(self.norm1.weight)
            nn.init.zeros_(self.norm1.bias)
        if self.use_norm and hasattr(self, "norm2"):
            nn.init.ones_(self.norm2.weight)
            nn.init.zeros_(self.norm2.bias)

    def forward(self, x, scale=None):
        cls_token = x[:, :5, :]
        x = x[:, 5:, :]

        if scale is None:
            scale = self.scale

        B, N, r = x.shape
        H = W = int(N**0.5)

        x_2d = x.permute(0, 2, 1).reshape(B, r, H, W).contiguous()

        x_scale1 = F.interpolate(
            x_2d,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )

        x_conv1 = self.dropout(self.act(self.conv1(x_scale1)))
        x_conv1 = F.interpolate(
            x_conv1, size=(H, W), mode="bilinear", align_corners=False
        )
        x_out1 = x_conv1

        x_scale2 = F.interpolate(
            x_2d, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        x_conv2 = self.dropout(self.act(self.conv2(x_scale2)))
        x_conv2 = F.interpolate(
            x_conv2, size=(H, W), mode="bilinear", align_corners=False
        )
        x_out2 = x_conv2

        if self.use_norm:
            x_out1 = x_out1.permute(0, 2, 3, 1).contiguous()
            x_out2 = x_out2.permute(0, 2, 3, 1).contiguous()
            x_out1 = self.norm1(x_out1)
            x_out2 = self.norm2(x_out2)
            x_out1 = x_out1.permute(0, 3, 1, 2).contiguous()
            x_out2 = x_out2.permute(0, 3, 1, 2).contiguous()

        x = (x_out1 + x_out2) / 2.0
        x = x.reshape(B, r, N).permute(0, 2, 1).contiguous()
        x = torch.cat([cls_token, x], dim=1)
        return x


class SingleConvExpert(nn.Module):
    def __init__(self, in_channels, out_channels, r, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = r
        self.dropout = nn.Dropout(dropout)

        self.proj_down = nn.Linear(in_channels, r, bias=False)
        self.act = nn.GELU()
        self.conv = LegacyConvExpert(r)
        self.proj_up = nn.Linear(r, out_channels, bias=False)
        self.scale = nn.Parameter(torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.proj_down.weight)
        nn.init.zeros_(self.proj_up.weight)

    def forward(self, x):
        x = self.dropout(self.act(self.proj_down(x)))
        x = self.dropout(self.act(self.conv(x)))
        x = self.proj_up(x) * self.scale
        return x


class DualMoeConvExpert(nn.Module):
    """
    简化版双尺度 MoE 卷积专家（无门控机制）

    优化点：
    1. 使用分组卷积减少参数量约 75%
    2. 使用 LayerNorm 提高训练稳定性
    3. 支持多种激活函数

    基于 DeepSeek-MoE、mixture-of-experts、shared/moe 的最佳实践
    """

    def __init__(
        self,
        r,
        kernel_size=3,
        groups=4,
        use_norm=True,
        activation="gelu",
    ):
        super().__init__()
        self.r = r
        self.kernel_size = kernel_size
        self.groups = groups
        self.use_norm = use_norm

        # 激活函数
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 分组卷积（减少参数量）
        self.conv1 = nn.Conv2d(
            r,
            r,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            r,
            r,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )

        # 归一化层（提高训练稳定性）
        if use_norm:
            self.norm1 = nn.LayerNorm(r)
            self.norm2 = nn.LayerNorm(r)

        # 初始化参数
        self._initialize_parameters()

    def _initialize_parameters(self):
        """参数初始化"""
        # 分组卷积使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.conv2.weight, a=math.sqrt(5))

        # 归一化层初始化
        if self.use_norm and hasattr(self, "norm1"):
            nn.init.ones_(self.norm1.weight)
            nn.init.zeros_(self.norm1.bias)
        if self.use_norm and hasattr(self, "norm2"):
            nn.init.ones_(self.norm2.weight)
            nn.init.zeros_(self.norm2.bias)

    def forward(self, x, scale):
        """
        Args:
            x: 输入张量，形状 (B, N, r) 或 (N, r)
            scale: 缩放因子

        Returns:
            output: 输出张量，形状与输入相同
        """

        B, N, r = x.shape
        H = W = int(N**0.5)

        # 转换为 2D 格式
        x_2d = x.permute(0, 2, 1).reshape(B, r, H, W).contiguous()

        # 上采样分支
        x_scaled1 = F.interpolate(
            x_2d, scale_factor=scale, mode="bilinear", align_corners=False
        )
        x_conv1 = self.conv1(x_scaled1)
        x_conv1 = self.act(x_conv1)
        x_out1 = F.interpolate(
            x_conv1, size=(H, W), mode="bilinear", align_corners=False
        )

        # 下采样分支
        x_scaled2 = F.interpolate(
            x_2d, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        x_conv2 = self.conv2(x_scaled2)
        x_conv2 = self.act(x_conv2)
        x_out2 = F.interpolate(
            x_conv2, size=(H, W), mode="bilinear", align_corners=False
        )

        # 归一化
        if self.use_norm:
            x_out1 = x_out1.permute(0, 2, 3, 1).contiguous()
            x_out2 = x_out2.permute(0, 2, 3, 1).contiguous()
            x_out1 = self.norm1(x_out1)
            x_out2 = self.norm2(x_out2)
            x_out1 = x_out1.permute(0, 3, 1, 2).contiguous()
            x_out2 = x_out2.permute(0, 3, 1, 2).contiguous()

        # 相加融合
        x_out = x_out1 + x_out2

        # 转换回原始格式
        x_out = x_out.reshape(B, r, N).permute(0, 2, 1).contiguous()

        return x_out
