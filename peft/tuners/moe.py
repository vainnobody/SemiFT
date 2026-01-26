import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class GatingNetwork(nn.Module):
    def __init__(
        self,
        r,
        num_experts,
        topk=1,
        loss_coef=1e-2,
        temperature=1.0,
        scoring_func="softmax",
        norm_topk_prob=True,
    ):
        super().__init__()
        self.r = r
        self.num_experts = num_experts
        self.topk = topk
        self.loss_coef = loss_coef
        self.temperature = temperature
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob

        self.gate_proj = nn.Linear(r, num_experts, bias=False)

        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))

    def forward(self, x):
        B, N, r = x.shape

        # Flatten tokens: (B, N, r) -> (B*N, r)
        # Use reshape to handle non-contiguous tensors
        flat_x = x.reshape(-1, r)

        # Compute gating scores for each token
        gate_logits = self.gate_proj(flat_x)

        # Apply softmax to get probabilities
        if self.scoring_func == "softmax":
            scores = torch.softmax(gate_logits / self.temperature, dim=-1)
        else:
            scores = gate_logits

        # Select top-k experts per token
        topk_weight, topk_idx = torch.topk(scores, k=self.topk, dim=-1, sorted=False)

        # Renormalize top-k weights if topk > 1
        if self.topk > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        # Reshape back to (B, N, topk)
        topk_weight = topk_weight.view(B, N, self.topk)
        topk_idx = topk_idx.view(B, N, self.topk)

        # Compute auxiliary loss for load balancing (DeepSeek-MoE style)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.training and self.loss_coef > 0.0:
            # Pi: Mean gating probability for each expert (importance)
            Pi = scores.mean(dim=0)

            # fi: Fraction of tokens routed to each expert (frequency)
            mask_ce = torch.nn.functional.one_hot(
                topk_idx.view(-1), num_classes=self.num_experts
            )
            ce = mask_ce.float().mean(0)
            fi = ce * self.num_experts

            # Compute load balancing loss: alpha * sum(Pi * fi)
            aux_loss = (Pi * fi).sum() * self.loss_coef

        return topk_idx, topk_weight, aux_loss


class SemiFt(nn.Module):
    def __init__(
        self, in_features, out_features, r, num_experts=4, scales=[1, 2, 4, 8], topk=1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.num_experts = num_experts
        self.scales = scales
        self.topk = topk

        self.experts = nn.ModuleList([nn.Linear(r, r) for _ in range(num_experts)])

        self.gating_network = GatingNetwork(
            r=r,
            num_experts=num_experts,
            topk=topk,
        )
        self.aux_loss = None

        self.proj_down = nn.Linear(in_features, r, bias=False)

        self.proj_up = nn.Linear(r, out_features, bias=False)

        self.act = nn.GELU()
        self.ls = LayerScale(out_features, init_values=1.0)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.proj_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.proj_up.weight)

    def forward(self, x):
        x = self.act(self.proj_down(x))

        # Extract special tokens (not routed through MoE)
        cls_token = x[:, :1, :]
        rel_token = x[:, 1:5, :]

        # Get patch tokens for MoE routing
        x = x[:, 5:, :]
        _x = x

        # Get routing decisions from gating network
        topk_idx, topk_weight, aux_loss = self.gating_network(x)

        # Flatten tokens for efficient routing: (B, N, r) -> (B*N, r)
        B, N, r = x.shape
        flat_x = x.reshape(-1, r)
        flat_topk_idx = topk_idx.reshape(-1)

        # Initialize combined features
        combine_features = torch.zeros_like(flat_x)

        # Process each expert on its assigned tokens
        if self.training:
            # Training mode: repeat tokens for each selected expert
            flat_x_expanded = flat_x.repeat_interleave(self.topk, dim=0)

            for i, expert in enumerate(self.experts):
                # Get tokens assigned to this expert
                expert_mask = flat_topk_idx == i
                if expert_mask.any():
                    expert_tokens = flat_x_expanded[expert_mask]
                    expert_out = expert(expert_tokens)

                    # Get weights for this expert's tokens
                    # Need to map back to original indices
                    expert_indices = torch.where(expert_mask)[0]
                    original_indices = expert_indices // self.topk
                    topk_positions = expert_indices % self.topk

                    # Get corresponding weights
                    flat_topk_weight = topk_weight.view(-1)
                    weights = flat_topk_weight[
                        original_indices * self.topk + topk_positions
                    ]

                    # Accumulate weighted outputs
                    combine_features[
                        original_indices
                    ] += expert_out * weights.unsqueeze(-1)

            # Reshape back to (B, N, r)
            combine_features = combine_features.view(B, N, r)
        else:
            # Inference mode: more efficient processing
            # Process each expert on tokens assigned to it
            for i, expert in enumerate(self.experts):
                # Find all (token_idx, topk_pos) pairs where this expert is selected
                expert_mask = flat_topk_idx == i
                if not expert_mask.any():
                    continue

                # Get original token indices (divide by topk to get original position)
                expert_flat_indices = torch.where(expert_mask)[0]
                original_token_indices = expert_flat_indices // self.topk

                # Get unique token indices for this expert
                unique_token_indices, inverse_indices = torch.unique(
                    original_token_indices, return_inverse=True
                )

                # Process unique tokens through this expert
                expert_tokens = flat_x[unique_token_indices]
                expert_out = expert(expert_tokens)

                # Get weights for each occurrence
                flat_topk_weight = topk_weight.view(-1)
                weights = flat_topk_weight[expert_flat_indices]

                # Accumulate weighted outputs using inverse indices
                for idx in range(len(unique_token_indices)):
                    token_idx = unique_token_indices[idx]
                    # Find all occurrences of this token for this expert
                    occurrences = original_token_indices == token_idx
                    occurrence_weights = weights[occurrences]
                    occurrence_outputs = (
                        expert_out[idx].unsqueeze(0).repeat(len(occurrence_weights), 1)
                    )

                    # Weighted sum
                    weighted_output = (
                        occurrence_outputs * occurrence_weights.unsqueeze(-1)
                    ).sum(dim=0)
                    combine_features[token_idx] += weighted_output

            # Reshape back to (B, N, r)
            combine_features = combine_features.view(B, N, r)

        # Add auxiliary loss to computation graph
        combine_features = AddAuxiliaryLoss.apply(combine_features, aux_loss)
        self.aux_loss = aux_loss

        # Final projection
        x = torch.cat([cls_token, rel_token, _x + combine_features], dim=1)
        x = self.ls(self.proj_up(x))

        return x


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # 1. 深度卷积 (Depthwise Convolution)
        # 每个通道使用独立的卷积核，groups = in_channels
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

        # 2. 逐点卷积 (Pointwise Convolution)
        # 使用 1x1 卷积核来融合深度卷积后的通道信息
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
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
