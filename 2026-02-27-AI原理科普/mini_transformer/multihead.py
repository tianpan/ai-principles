"""
Mini Transformer - Multi-Head Attention

模块作用：并行运行多个独立的 attention，每个关注不同的模式

代码位置 → 模块对应：
- W_q, W_k, W_v: 一次性计算所有 head 的 Q/K/V
- view + transpose: 将 d_model 分割成 num_heads 个 d_k
- heads.concat @ W_o: 拼接所有 head 并投影
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 层

    Args:
        d_model: 模型维度
        num_heads: 注意力头数量

    Input:
        x: [batch_size, seq_len, d_model]
        mask: 可选的注意力掩码

    Output:
        output: [batch_size, seq_len, d_model]
        attention_weights: [batch_size, num_heads, seq_len, seq_len]
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个 head 的维度

        # Q, K, V 投影（一次性计算所有 head）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量, shape [batch_size, seq_len, d_model]
            mask: 注意力掩码

        Returns:
            output: 多头注意力输出
            attention_weights: 所有 head 的注意力权重
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: 线性投影
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: 分割成多个头
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: 计算 attention (每个 head 独立)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)

        # Step 4: 加权求和
        heads = torch.matmul(attention_weights, V)  # [batch, num_heads, seq_len, d_k]

        # Step 5: 拼接所有 head
        # [batch, num_heads, seq_len, d_k] -> [batch, seq_len, num_heads, d_k] -> [batch, seq_len, d_model]
        concat = heads.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Step 6: 最终投影
        output = self.W_o(concat)

        return output, attention_weights


if __name__ == "__main__":
    # 测试
    d_model = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    output, weights = mha(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    assert output.shape == x.shape
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    print("✓ MultiHeadAttention 测试通过!")
