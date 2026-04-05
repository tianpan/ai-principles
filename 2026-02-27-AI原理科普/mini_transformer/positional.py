"""
Mini Transformer - Positional Encoding

模块作用：为序列注入位置信息，弥补 Self-Attention 对位置不敏感的特性

代码位置 → 模块对应：
- sin/cos 计算: 生成不同频率的波形
- register_buffer: 不参与训练的固定参数
- forward: 将位置编码加到 embedding 上
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding 层 (sin/cos 编码)

    Args:
        d_model: embedding 维度
        max_len: 最大序列长度

    Input:
        x: [batch_size, seq_len, d_model] - token embeddings

    Output:
        [batch_size, seq_len, d_model] - 加上位置编码后的 embeddings
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母项: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加 batch 维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为 buffer（不参与训练，但会保存到模型中）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: token embeddings, shape [batch_size, seq_len, d_model]

        Returns:
            加上位置编码后的 embeddings
        """
        # 只取需要的长度
        return x + self.pe[:, :x.size(1)]


if __name__ == "__main__":
    # 测试
    d_model = 64
    batch_size = 2
    seq_len = 10

    pos_enc = PositionalEncoding(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    output = pos_enc(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape
    print("✓ PositionalEncoding 测试通过!")

    # 验证位置编码值
    print(f"\n位置 0 的前 4 维: {pos_enc.pe[0, 0, :4].tolist()}")
    print(f"位置 1 的前 4 维: {pos_enc.pe[0, 1, :4].tolist()}")
