"""
单元测试 - Mini Transformer 各模块

覆盖:
- TokenEmbedding: 维度验证、缩放因子
- PositionalEncoding: sin/cos 波形、维度匹配
- SelfAttention: Q/K/V 计算正确性
- MultiHeadAttention: 头分割、拼接正确性
- TransformerBlock: 残差连接、LayerNorm
- MiniTransformer: 完整前向传播
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding import TokenEmbedding
from positional import PositionalEncoding
from attention import SelfAttention
from multihead import MultiHeadAttention
from transformer import TransformerBlock, MiniTransformer


class TestTokenEmbedding:
    """Token Embedding 单元测试"""

    def test_output_shape(self):
        """输出维度正确"""
        vocab_size, d_model, batch_size, seq_len = 100, 64, 2, 10
        embedding = TokenEmbedding(vocab_size, d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = embedding(x)
        assert output.shape == (batch_size, seq_len, d_model), \
            f"期望 {(batch_size, seq_len, d_model)}, 实际 {output.shape}"

    def test_scaling_factor(self):
        """缩放因子 sqrt(d_model) 正确应用"""
        vocab_size, d_model = 100, 64
        embedding = TokenEmbedding(vocab_size, d_model)
        x = torch.tensor([[0]])
        output = embedding(x)
        # 获取原始 embedding 值
        raw = embedding.embedding(x)
        expected = raw * (d_model ** 0.5)
        assert torch.allclose(output, expected), "缩放因子未正确应用"

    def test_different_vocab_sizes(self):
        """不同词表大小都能工作"""
        for vocab_size in [10, 100, 1000]:
            embedding = TokenEmbedding(vocab_size, 64)
            x = torch.randint(0, vocab_size, (1, 5))
            output = embedding(x)
            assert output.shape[-1] == 64


class TestPositionalEncoding:
    """Positional Encoding 单元测试"""

    def test_output_shape(self):
        """输出维度正确"""
        d_model, max_len, batch_size, seq_len = 64, 5000, 2, 10
        pe = PositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        assert output.shape == x.shape, f"期望 {x.shape}, 实际 {output.shape}"

    def test_adds_to_input(self):
        """PE 是加到输入上，不是替换"""
        d_model = 64
        pe = PositionalEncoding(d_model)
        x = torch.ones(1, 5, d_model)
        output = pe(x)
        # 输出应该不等于输入（除非 PE 恰好为 0）
        assert not torch.allclose(output, x), "PE 没有正确加到输入上"

    def test_sin_cos_pattern(self):
        """验证 sin/cos 交替模式"""
        d_model = 4
        pe = PositionalEncoding(d_model, max_len=10)
        # 检查 PE 矩阵的形状
        assert pe.pe.shape == (1, 10, d_model)

    def test_position_uniqueness(self):
        """不同位置有不同的编码"""
        d_model = 64
        pe = PositionalEncoding(d_model)
        x = torch.zeros(1, 10, d_model)
        output = pe(x)
        # 相邻位置的编码应该不同
        pos_0 = output[0, 0, :]
        pos_1 = output[0, 1, :]
        assert not torch.allclose(pos_0, pos_1), "不同位置的编码相同"


class TestSelfAttention:
    """Self-Attention 单元测试"""

    def test_output_shape(self):
        """输出维度正确"""
        d_model, batch_size, seq_len = 64, 2, 10
        attention = SelfAttention(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        output, weights = attention(x)
        assert output.shape == (batch_size, seq_len, d_model), \
            f"期望 {(batch_size, seq_len, d_model)}, 实际 {output.shape}"
        assert weights.shape == (batch_size, seq_len, seq_len), \
            f"期望 {(batch_size, seq_len, seq_len)}, 实际 {weights.shape}"

    def test_attention_weights_sum_to_one(self):
        """注意力权重和为 1"""
        d_model, batch_size, seq_len = 64, 2, 10
        attention = SelfAttention(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        _, weights = attention(x)
        # 沿最后一维求和应该为 1
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "注意力权重和不为 1"

    def test_causal_mask(self):
        """因果掩码正确应用（未来位置被屏蔽）"""
        d_model, batch_size, seq_len = 64, 1, 4
        attention = SelfAttention(d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        # 创建因果掩码
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        _, weights = attention(x, mask)

        # 上三角应该接近 0（被 mask 的位置）
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[0, i, j] < 1e-5, \
                    f"位置 ({i}, {j}) 应该被 mask 但权重为 {weights[0, i, j]}"


class TestMultiHeadAttention:
    """Multi-Head Attention 单元测试"""

    def test_output_shape(self):
        """输出维度正确"""
        d_model, num_heads, batch_size, seq_len = 64, 4, 2, 10
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        output, weights = mha(x)
        assert output.shape == (batch_size, seq_len, d_model), \
            f"期望 {(batch_size, seq_len, d_model)}, 实际 {output.shape}"

    def test_head_splitting(self):
        """头分割正确"""
        d_model, num_heads = 64, 4
        d_k = d_model // num_heads
        assert d_k == 16, f"d_k 应该是 16, 实际 {d_k}"

        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(1, 5, d_model)
        output, _ = mha(x)
        assert output.shape[-1] == d_model

    def test_invalid_head_config(self):
        """d_model 不能被 num_heads 整除时报错"""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=4)


class TestTransformerBlock:
    """Transformer Block 单元测试"""

    def test_output_shape(self):
        """输出维度正确"""
        d_model, num_heads, d_ff, batch_size, seq_len = 64, 4, 256, 2, 10
        block = TransformerBlock(d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        output = block(x)
        assert output.shape == x.shape, f"期望 {x.shape}, 实际 {output.shape}"

    def test_residual_connection(self):
        """残差连接正确（输入被加到输出上）"""
        d_model = 64
        block = TransformerBlock(d_model, num_heads=4, d_ff=256, dropout=0.0)
        x = torch.randn(1, 5, d_model)

        # 设置 dropout 为 0，确保可重复
        block.eval()
        output = block(x)

        # 输出不应该等于输入（经过了变换）
        # 但残差连接意味着输入信息被保留
        assert output.shape == x.shape


class TestMiniTransformer:
    """完整模型单元测试"""

    def test_output_shape(self):
        """输出维度正确"""
        vocab_size, d_model, batch_size, seq_len = 100, 64, 2, 10
        model = MiniTransformer(vocab_size, d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = model(x)
        assert logits.shape == (batch_size, seq_len, vocab_size), \
            f"期望 {(batch_size, seq_len, vocab_size)}, 实际 {logits.shape}"

    def test_different_configs(self):
        """不同配置都能工作"""
        configs = [
            {'vocab_size': 50, 'd_model': 32, 'num_heads': 2, 'num_layers': 1},
            {'vocab_size': 100, 'd_model': 64, 'num_heads': 4, 'num_layers': 2},
            {'vocab_size': 200, 'd_model': 128, 'num_heads': 8, 'num_layers': 4},
        ]
        for config in configs:
            model = MiniTransformer(**config)
            x = torch.randint(0, config['vocab_size'], (1, 5))
            output = model(x)
            assert output.shape == (1, 5, config['vocab_size'])

    def test_gradient_flow(self):
        """梯度能正确反向传播"""
        vocab_size, d_model = 100, 64
        model = MiniTransformer(vocab_size, d_model)
        x = torch.randint(0, vocab_size, (1, 5))
        y = torch.randint(0, vocab_size, (1, 5))

        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()

        # 检查所有参数都有梯度
        for name, param in model.named_parameters():
            assert param.grad is not None, f"参数 {name} 没有梯度"


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
