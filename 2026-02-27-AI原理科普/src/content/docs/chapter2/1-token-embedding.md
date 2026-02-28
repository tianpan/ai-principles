---
title: Token 与 Embedding
description: 理解文本如何转换为向量表示
---

import TerminologyCard from '@components/common/TerminologyCard.astro';

# 2.1 Token 与 Embedding

在 Transformer 中，文本首先要被转换为数字表示。这个过程分为两步：
1. **Tokenization**：将文本切分为 token
2. **Embedding**：将 token 映射为向量

## 什么是 Token？

<TerminologyCard term="Token" definition="文本处理的基本单位。可以是单词、子词或字符。" example="'hello world' 可能被分为 ['hello', ' world'] 两个 token" />

### Tokenizer 的工作方式

不同的 tokenizer 有不同的切分策略：

| 类型 | 示例 | 特点 |
|------|------|------|
| Word-level | ['hello', 'world'] | 简单但词表大 |
| Character-level | ['h','e','l','l','o'] | 词表小但序列长 |
| Subword (BPE) | ['hel', 'lo', 'world'] | 平衡词表和序列长度 |

现代 LLM 大多使用 **BPE (Byte Pair Encoding)** 或 **SentencePiece**。

## 什么是 Embedding？

<TerminologyCard term="Embedding" definition="将离散的 token 映射到连续向量空间。" example="token 'hello' → [0.1, -0.3, 0.7, ...] (d_model 维向量)" />

### Embedding 的直觉

想象一个"语义空间"：
- 相似含义的词，向量也相似
- 可以做"语义运算"：`king - man + woman ≈ queen`

### 数学表示

给定词表大小 $V$ 和 embedding 维度 $d$：

```
Embedding 矩阵: E ∈ ℝ^(V × d)

第 i 个 token 的 embedding: e_i = E[i]
```

## 代码实现

```python
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # 查表操作：token id → 向量
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len] (token ids)
        # return: [batch_size, seq_len, d_model]
        return self.embedding(x)
```

## 小结

- **Token** 是文本处理的基本单位
- **Embedding** 将 token 映射为向量，捕获语义信息
- 通过查表操作实现，高效且可学习

---

[下一节：Self-Attention →](/chapter2/2-self-attention)
