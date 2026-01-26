import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedVisualExpert(nn.Module):
    """
    针对 Token 级路由优化的视觉专家 (Semantic Segmentation Optimized)

    特点：
    1. 纯 Token 处理：不依赖空间结构还原，适应稀疏输入 (N, C)。
    2. 双流设计：结合了 MLP 的特征变换能力和 Depthwise Conv 的归纳偏置。
    3. 门控机制：引入 Gating 机制提升非线性表达能力（语义分割关键）。
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        kernel_size=3,  # 1D 卷积核，用于增强特征混合
        act_layer=nn.SiLU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4  # 通常膨胀4倍

        # 1. 投影层 (对应 1x1 Conv)
        # 将输入映射到双倍维度，一半用于 Gate，一半用于 Value
        self.fc1 = nn.Linear(in_features, hidden_features * 2)

        # 2. 深度可分离卷积 (Depthwise Conv1d)
        # 即使 Token 是稀疏的，DW-Conv 也能作为一种 Learnable Position Mixing
        # 或者仅仅作为增强的通道混合器。
        # groups=hidden_features 意味着每个通道独立卷积，参数量极小但效果好
        self.dwconv = nn.Conv1d(
            hidden_features,
            hidden_features,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=hidden_features,
            bias=True,
        )

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        # 归一化，通常在专家内部做一次 Norm 有助于训练稳定
        self.norm = nn.LayerNorm(in_features)

        self._init_weights()

    def _init_weights(self):
        # 优化初始化策略
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, scale=None):
        """
        Args:
            x: 输入张量，形状 (N, C) 或 (B, N, C)。
               如果是 Token 级路由，通常是 (Total_Tokens, C)。
            scale: 在此设计中不再强制需要，保留接口兼容性。
        """
        # 记录原始形状
        is_batched = x.dim() == 3
        if is_batched:
            B, N, C = x.shape
            x = x.reshape(B * N, C)

        # 1. Pre-Norm (ResNet/Transformer 标准做法)
        shortcut = x
        x = self.norm(x)

        # 2. 投影与双流分割 (Dimension Expansion)
        # x_proj shape: (N, 2 * hidden)
        x_proj = self.fc1(x)

        # 将特征分为两部分：Gate流 和 Value流
        x_gate, x_value = x_proj.chunk(2, dim=-1)

        # 3. 处理 Value 流 (模拟视觉感知)
        # 即使是 Token 列表，转置后做 1D 卷积也能捕捉特征间的局部相关性
        # (N, C) -> (1, C, N) -> Conv -> (1, C, N) -> (N, C)
        # 这里的 unsqueeze(0) 是为了利用 Conv1d API，视为 Batch=1 的长序列
        x_value = x_value.unsqueeze(0).transpose(1, 2)  # -> (1, Hidden, N)
        x_value = self.dwconv(x_value)
        x_value = x_value.transpose(1, 2).squeeze(0)  # -> (N, Hidden)

        # 4. 门控激活 (Gating / SwiGLU 变体)
        # Output = Act(Gate) * Conv(Value)
        # 这种机制允许模型动态选择通过哪些特征，对分割边缘极其重要
        x = self.act(x_gate) * x_value

        # 5. 最终投影与残差连接
        x = self.drop(x)
        x = self.fc2(x)
        x = x + shortcut

        # 恢复形状
        if is_batched:
            x = x.reshape(B, N, C)

        return x
