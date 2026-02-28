---
title: Multi-Head Attention
description: 理解多头注意力机制
---

import TerminologyCard from '@components/common/TerminologyCard.astro';

# 2.3 Multi-Head Attention

多头注意力让模型同时从**多个角度**"观察"序列。

## 为什么需要多头？

<TerminologyCard term="Multi-Head Attention" definition="并行运行多个独立的 attention，每个关注不同的模式。" />

单个 attention 头可能只能捕获一种关系：
- 头 1：关注语法关系（主语-谓语）
- 头 2：关注语义关系（同义词）
- 头 3：关注位置关系（相邻词）

多头让模型同时学习多种模式。

## 计算过程

### Step 1: 分割成多个头

将 d_model 维度分成 h 个头，每个头维度 d_k = d_model / h

```
Q, K, V: [batch, seq_len, d_model]
→ 分割为 h 个头
Q_i, K_i, V_i: [batch, seq_len, d_k]
```

### Step 2: 每个 head 独立计算 attention

```
head_i = Attention(Q_i, K_i, V_i)
```

### Step 3: 拼接所有 head

```
concat = [head_1, head_2, ..., head_h]  # [batch, seq_len, d_model]
```

### Step 4: 最终投影

```
output = concat @ W_O
```

## 完整公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

其中：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

## 代码实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q、K、V 投影（一次性计算所有 head）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. 分割成多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 现在: [batch, num_heads, seq_len, d_k]

        # 3. 计算 attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        heads = torch.matmul(attention_weights, V)

        # 4. 拼接所有 head
        concat = heads.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 5. 最终投影
        output = self.W_o(concat)

        return output, attention_weights
```

## 典型配置

| 模型 | d_model | num_heads | d_k |
|------|---------|-----------|-----|
| BERT-Base | 768 | 12 | 64 |
| GPT-2 | 768 | 12 | 64 |
| LLaMA-7B | 4096 | 32 | 128 |

## 小结

- 多头 = 多个独立的 attention 并行计算
- 每个 head 关注不同的模式
- 最后拼接并通过线性层整合

---

[下一节：Positional Encoding →](/chapter2/4-positional-encoding)
