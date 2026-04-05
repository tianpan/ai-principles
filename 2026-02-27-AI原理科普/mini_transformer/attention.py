"""
Mini Transformer - Self-Attention

模块作用：让序列中的每个元素都能关注到其他所有元素，计算它们之间的相关性

代码位置 → 模块对应：
- W_q, W_k, W_v: Q/K/V 投影矩阵
- scores = Q @ K^T / √d_k: 计算注意力分数
- softmax: 归一化为概率分布
- output = weights @ V: 加权求和
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class SelfAttention(nn.Module):
    """
    Self-Attention 层

    Args:
        d_model: 模型维度

    Input:
        x: [batch_size, seq_len, d_model]
        mask: 可选的注意力掩码

    Output:
        output: [batch_size, seq_len, d_model]
        attention_weights: [batch_size, seq_len, seq_len]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_k = d_model

        # Q, K, V 投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量, shape [batch_size, seq_len, d_model]
            mask: 注意力掩码, shape [batch_size, seq_len, seq_len]

        Returns:
            output: 注意力输出
            attention_weights: 注意力权重矩阵
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: 计算 Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: 计算注意力分数 Q @ K^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Step 3: 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 4: Softmax 归一化
        attention_weights = torch.softmax(scores, dim=-1)

        # Step 5: 加权求和
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


if __name__ == "__main__":
    # 测试
    d_model = 64
    batch_size = 2
    seq_len = 10

    attention = SelfAttention(d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    output, weights = attention(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    assert output.shape == x.shape
    assert weights.shape == (batch_size, seq_len, seq_len)

    # 验证权重和为 1
    weight_sum = weights.sum(dim=-1)
    print(f"权重和 (应该接近 1): {weight_sum[0, 0].item():.4f}")

    print("✓ SelfAttention 测试通过!")
