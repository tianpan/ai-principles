"""
Mini Transformer - Complete Model

完整的 Transformer 模型，包含：
- Token Embedding
- Positional Encoding
- N 个 Transformer Block (Multi-Head Attention + FFN + LayerNorm + Residual)
- 输出层

代码位置 → 模块对应：
- token_embedding: 将 token id 转为向量
- pos_encoding: 加入位置信息
- blocks: 多层 Transformer Block
- norm: 最终的 LayerNorm
- output: 映射到词表
"""

import torch
import torch.nn as nn
from typing import Optional

from embedding import TokenEmbedding
from positional import PositionalEncoding
from multihead import MultiHeadAttention


class FeedForward(nn.Module):
    """
    前馈神经网络

    Args:
        d_model: 模型维度
        d_ff: 隐藏层维度 (通常是 d_model 的 4 倍)
        dropout: dropout 比率
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    单个 Transformer Block

    结构 (Pre-LN):
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN Attention
        attn_out, _ = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)

        # Pre-LN FFN
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)

        return x


class MiniTransformer(nn.Module):
    """
    完整的 Mini Transformer 模型

    Args:
        vocab_size: 词表大小
        d_model: 模型维度
        num_heads: 注意力头数量
        num_layers: Transformer 层数
        d_ff: FFN 隐藏层维度
        max_len: 最大序列长度
        dropout: dropout 比率
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        max_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Embedding 层
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最终 LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # 输出层 (语言模型头)
        self.output = nn.Linear(d_model, vocab_size)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: token IDs, shape [batch_size, seq_len]
            mask: 注意力掩码

        Returns:
            logits: shape [batch_size, seq_len, vocab_size]
        """
        # Embedding
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)

        # 最终 LayerNorm
        x = self.norm(x)

        # 输出 logits
        logits = self.output(x)

        return logits


if __name__ == "__main__":
    # 测试
    vocab_size = 1000
    batch_size = 2
    seq_len = 32

    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("✓ MiniTransformer 测试通过!")
