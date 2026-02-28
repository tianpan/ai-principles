"""
Mini Transformer - Token Embedding Layer

模块作用：将离散的 token ID 映射为连续的向量表示

代码位置 → 模块对应：
- nn.Embedding: 查表操作，将 token id 转换为向量
- forward: 前向传播，返回 embedding 向量
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token Embedding 层

    Args:
        vocab_size: 词表大小
        d_model: embedding 维度

    Input:
        x: [batch_size, seq_len] - token IDs

    Output:
        [batch_size, seq_len, d_model] - embedding vectors
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: token IDs, shape [batch_size, seq_len]

        Returns:
            embedding vectors, shape [batch_size, seq_len, d_model]
        """
        return self.embedding(x) * (self.d_model ** 0.5)  # 缩放因子


if __name__ == "__main__":
    # 测试
    vocab_size = 1000
    d_model = 64
    batch_size = 2
    seq_len = 10

    embedding = TokenEmbedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = embedding(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ TokenEmbedding 测试通过!")
