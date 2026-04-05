---
title: FFN 与输出层
description: 理解前馈网络和输出层
---

import TerminologyCard from '@components/common/TerminologyCard.astro';

# 2.6 FFN 与输出层

前馈网络（FFN）为 Transformer 提供**非线性变换**能力。

## Feed-Forward Network (FFN)

<TerminologyCard term="FFN" definition="两个线性变换加一个激活函数，对每个位置独立作用。" />

### 结构

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

通常：
- W₁: d_model → d_ff（扩展，通常 d_ff = 4 × d_model）
- W₂: d_ff → d_model（压缩回原维度）

### 为什么需要 FFN？

Self-Attention 负责**信息聚合**，FFN 负责**信息处理**：
- Attention：让 token 之间交流
- FFN：对每个 token 的信息进行加工

类比：
- Attention = 讨论会议（大家交流信息）
- FFN = 独立思考（每个人整理自己的笔记）

### 激活函数

| 激活函数 | 使用模型 |
|----------|----------|
| ReLU | 原始 Transformer |
| GELU | BERT, GPT, LLaMA |
| Swish/GLU | PaLM, LLaMA 2 |

## 输出层

经过多层 Transformer Block 后，需要将 hidden state 映射回词表。

### Logits 计算

```python
logits = hidden @ W_vocab  # [batch, seq_len, vocab_size]
```

### Softmax 得到概率

```python
probs = softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
```

### 语言模型头

通常有两种方式：
1. **共享权重**：W_vocab = Embedding^T
2. **独立权重**：单独的线性层

## 完整的 Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-LN Attention
        x = x + self.attention(self.norm1(x), mask)
        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        return x
```

## 完整的 Transformer 模型

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=512):
        super().__init__()

        # Embedding + Positional Encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # 最终 LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # 输出层（共享权重）
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, x, mask=None):
        # Embedding
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)

        # 输出
        x = self.norm(x)
        logits = self.output(x)

        return logits
```

## 小结

- **FFN**：提供非线性变换，通常扩展 4 倍再压缩
- **输出层**：将 hidden state 映射回词表概率
- **共享权重**：减少参数量，常用于小型模型

---

🎉 **恭喜！** 你已经学习了 Transformer 的所有核心组件。

[前往 Mini Transformer Lab →](/lab/intro)
