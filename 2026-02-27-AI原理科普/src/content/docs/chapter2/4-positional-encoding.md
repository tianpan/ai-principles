---
title: Positional Encoding
description: 理解位置编码
---

import TerminologyCard from '@components/common/TerminologyCard.astro';

# 2.4 Positional Encoding

位置编码为序列注入**位置信息**，弥补 Self-Attention 对位置不敏感的特性。

## 问题：Self-Attention 不知道位置

<TerminologyCard term="排列等变性" definition="打乱输入顺序，输出也会对应打乱，但数值不变。Self-Attention 具有这个性质。" />

对于 Self-Attention：
- 输入 ["A", "B", "C"] 和 ["C", "B", "A"]
- 除了顺序，每个 token 的计算结果完全一样

这意味着模型**不知道** token 的位置！

## 解决方案：位置编码

给每个位置一个唯一的向量，加到 embedding 上：

```
input = token_embedding + positional_encoding
```

## sin/cos 编码（原始 Transformer）

<TerminologyCard term="sin/cos 位置编码" definition="使用不同频率的正弦和余弦函数生成位置向量。" />

### 公式

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

其中：
- pos：位置索引
- i：维度索引
- d：embedding 维度

### 直觉理解

- 不同维度使用不同频率的波形
- 低位维度（i 小）：高频，快速变化
- 高位维度（i 大）：低频，缓慢变化
- 组合起来可以唯一表示每个位置

### 为什么用 sin/cos？

1. **有界**：值在 [-1, 1] 之间
2. **可外推**：可以处理训练时没见过的长度
3. **相对位置**：PE(pos+k) 可以用 PE(pos) 的线性变换表示

## 代码实现

```python
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母项
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer（不参与训练）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # 加上位置编码
        return x + self.pe[:, :x.size(1)]
```

## 其他位置编码方式

| 方法 | 特点 |
|------|------|
| Learnable (GPT) | 可学习的位置 embedding |
| RoPE (LLaMA) | 旋转位置编码，相对位置 |
| ALiBi | 线性偏置，外推能力强 |

## 小结

- Self-Attention 本身不知道位置
- 位置编码将位置信息注入到 embedding
- sin/cos 编码是经典方案，现代模型有更多选择

---

[下一节：残差连接与 LayerNorm →](/chapter2/5-residual-layernorm)
