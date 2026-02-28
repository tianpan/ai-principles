---
title: 残差连接与 LayerNorm
description: 理解残差连接和层归一化
---

import TerminologyCard from '@components/common/TerminologyCard.astro';

# 2.5 残差连接与 LayerNorm

残差连接和层归一化是 Transformer 训练**稳定性**的关键。

## 残差连接（Residual Connection）

<TerminologyCard term="残差连接" definition="将层的输入直接加到输出上：output = x + sublayer(x)" />

### 为什么需要？

深层网络面临**梯度消失**问题：反向传播时，梯度需要经过很多层，越来越小。

残差连接提供了一条"高速公路"：
```
梯度可以直接通过 x 传回，不需要经过 sublayer
```

### Transformer 中的应用

每个子层都有残差连接：
```
x = x + MultiHeadAttention(x)
x = x + FFN(x)
```

## Layer Normalization

<TerminologyCard term="LayerNorm" definition="对每个样本的所有特征进行归一化，使均值为 0，方差为 1。" />

### 公式

$$\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

其中：
- μ：该样本所有特征的均值
- σ：标准差
- γ、β：可学习的缩放和偏移参数

### 为什么用 LayerNorm 而不是 BatchNorm？

| | BatchNorm | LayerNorm |
|---|-----------|-----------|
| 归一化维度 | 跨 batch | 跨特征 |
| 对 batch size 敏感 | 是 | 否 |
| 适合序列模型 | 不太适合 | 非常适合 |
| 推理时行为 | 需要统计量 | 不需要 |

### Transformer 中的顺序

原始 Transformer（Post-LN）：
```
x = LayerNorm(x + Sublayer(x))
```

现代模型（Pre-LN，更稳定）：
```
x = x + Sublayer(LayerNorm(x))
```

## 代码实现

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN 结构
        # 1. Attention 子层
        attn_output, _ = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_output)

        # 2. FFN 子层
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)

        return x
```

## 小结

- **残差连接**：让梯度更容易流动，支持深层网络训练
- **LayerNorm**：稳定每层的输入分布，加速训练
- **Pre-LN**：现代模型首选，训练更稳定

---

[下一节：FFN 与输出层 →](/chapter2/6-ffn-output)
