---
title: Self-Attention
description: 理解自注意力机制
---

import TerminologyCard from '@components/common/TerminologyCard.astro';

# 2.2 Self-Attention

Self-Attention 是 Transformer 的**核心创新**，让模型理解序列内部的关系。

## 核心思想

<TerminologyCard term="Self-Attention" definition="让序列中的每个元素都能关注到其他所有元素，计算它们之间的相关性。" />

想象你在读一句话："The **bank** of the river"：
- 要理解 "bank" 的含义，你需要关注 "river"
- Self-Attention 就是让模型学会这种"关注"

## Q、K、V：查询、键、值

<TerminologyCard term="Query (Q)" definition="我在找什么？每个 token 发出的查询向量。" />
<TerminologyCard term="Key (K)" definition="我是什么？每个 token 的标识向量。" />
<TerminologyCard term="Value (V)" definition="我的内容是什么？每个 token 的信息向量。" />

### 类比理解

想象在图书馆找书：
- **Query**：你的需求（"我想找关于 AI 的书"）
- **Key**：书的标签/分类
- **Value**：书的实际内容

Q 和 K 的匹配程度决定你"关注"这本书多少。

## 计算步骤

### Step 1: 生成 Q、K、V

```
Q = X @ W_Q    # X 是输入 embedding
K = X @ W_K
V = X @ W_V
```

### Step 2: 计算注意力分数

```
scores = Q @ K^T / √d_k
```

为什么要除以 √d_k？防止点积过大导致 softmax 梯度消失。

### Step 3: Softmax 归一化

```
attention_weights = softmax(scores)
```

### Step 4: 加权求和

```
output = attention_weights @ V
```

## 完整公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

## 代码实现

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_k = d_model

        # Q、K、V 投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch, seq_len, d_model]

        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
```

## 小结

- Self-Attention 让每个 token 都能"看到"其他所有 token
- 通过 Q、K、V 三个投影实现
- 注意力权重表示 token 之间的相关性

---

[下一节：Multi-Head Attention →](/chapter2/3-multi-head-attention)
