---
title: 代码实现
description: 从零实现 Mini Transformer
---

# 代码实现

本章将逐步实现 Mini Transformer 的各个模块。每个模块都遵循 **概念 → 代码 → 验证** 的学习闭环。

## 模块映射

| 概念章节 | 代码模块 | 文件 |
|---------|---------|------|
| 2.1 Token/Embedding | TokenEmbedding | `embedding.py` |
| 2.4 位置编码 | PositionalEncoding | `positional.py` |
| 2.2 Self-Attention | SelfAttention | `attention.py` |
| 2.3 Multi-Head | MultiHeadAttention | `multihead.py` |
| 2.5 残差+LayerNorm | TransformerBlock | `transformer.py` |
| 2.6 FFN+输出 | MiniTransformer | `transformer.py` |

---

## 1. Token Embedding

**概念回顾**：将离散的 token ID 转换为连续的向量表示。

```python
# embedding.py
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len] -> [batch, seq_len, d_model]
        return self.embedding(x) * (self.d_model ** 0.5)
```

**验证**：
```python
>>> emb = TokenEmbedding(vocab_size=100, d_model=64)
>>> x = torch.randint(0, 100, (2, 10))  # batch=2, seq_len=10
>>> emb(x).shape
torch.Size([2, 10, 64])  # ✓ 输出维度正确
```

---

## 2. Positional Encoding

**概念回顾**：使用 sin/cos 函数为序列注入位置信息。

```python
# positional.py
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                           * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]
```

**验证**：
```python
>>> pe = PositionalEncoding(d_model=64)
>>> x = torch.randn(1, 10, 64)
>>> pe(x).shape
torch.Size([1, 10, 64])  # ✓ 保持维度不变
```

---

## 3. Self-Attention

**概念回顾**：Q·Kᵀ 计算注意力分数，加权求和 V。

```python
# attention.py
class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.d_k = d_model

    def forward(self, x, mask=None):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V), weights
```

**验证**：
```python
>>> attn = SelfAttention(d_model=64)
>>> x = torch.randn(1, 5, 64)
>>> out, weights = attn(x)
>>> weights.sum(dim=-1)  # 注意力权重和为 1
tensor([[1., 1., 1., 1., 1.]])  # ✓
```

---

## 4. Multi-Head Attention

**概念回顾**：将 d_model 分割成多个头并行计算。

```python
# multihead.py
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        Q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        heads = torch.matmul(weights, V)
        concat = heads.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(concat), weights
```

**验证**：
```python
>>> mha = MultiHeadAttention(d_model=64, num_heads=4)
>>> x = torch.randn(2, 10, 64)
>>> out, _ = mha(x)
>>> out.shape
torch.Size([2, 10, 64])  # ✓ 输出维度正确
```

---

## 5. 完整模型

将所有模块组合成完整的 Mini Transformer：

```python
# transformer.py
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4,
                 num_layers=2, d_ff=256, max_len=128):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.output(self.norm(x))
```

---

## 下一步

完成了代码实现？继续 [训练与推理](/lab/training) 了解如何训练模型。
