"""
集成测试 - Mini Transformer 训练流水线

覆盖:
- CharTokenizer: 编码解码一致性
- 数据加载: 数据集创建、批次生成
- 训练流程: loss 下降、梯度更新
- 推理流程: 生成 token 数量
- 端到端: 训练 → 保存 → 加载 → 生成
"""

import pytest
import torch
import torch.nn as nn
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import CharTokenizer, get_dataset, SHAKESPEARE_TEXT, DIALOGUE_TEXT
from transformer import MiniTransformer
from train import TextDataset, train
from generate import generate


class TestCharTokenizer:
    """Tokenizer 集成测试"""

    def test_encode_decode_consistency(self):
        """编码后解码应得到原文"""
        text = "To be, or not to be"
        tokenizer = CharTokenizer(text)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"编码解码不一致: '{decoded}' != '{text}'"

    def test_vocab_from_text(self):
        """词表从文本正确构建"""
        text = "hello"
        tokenizer = CharTokenizer(text)
        assert tokenizer.vocab_size == 4, f"词表大小应为 4, 实际 {tokenizer.vocab_size}"
        assert set(tokenizer.chars) == {'e', 'h', 'l', 'o'}

    def test_unknown_char_handling(self):
        """未知字符应该报错"""
        text = "abc"
        tokenizer = CharTokenizer(text)
        with pytest.raises(KeyError):
            tokenizer.encode("xyz")


class TestDataset:
    """数据集集成测试"""

    def test_dataset_creation(self):
        """数据集正确创建"""
        text, tokenizer, data = get_dataset('shakespeare')
        dataset = TextDataset(data, seq_len=32)
        assert len(dataset) > 0, "数据集为空"

    def test_batch_generation(self):
        """批次生成正确"""
        from torch.utils.data import DataLoader

        text, tokenizer, data = get_dataset('shakespeare')
        dataset = TextDataset(data, seq_len=32)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        x, y = next(iter(dataloader))
        assert x.shape == (4, 32), f"输入形状错误: {x.shape}"
        assert y.shape == (4, 32), f"目标形状错误: {y.shape}"

    def test_next_token_prediction(self):
        """目标是下一个 token"""
        data = list(range(100))
        dataset = TextDataset(data, seq_len=10)

        x, y = dataset[0]
        # y 应该是 x 向右移一位
        assert torch.equal(y, torch.tensor(data[1:11])), "目标不是下一个 token"


class TestTrainingPipeline:
    """训练流水线集成测试"""

    @pytest.fixture
    def setup_training(self):
        """设置训练环境"""
        text, tokenizer, data = get_dataset('shakespeare')
        vocab_size = tokenizer.vocab_size

        model = MiniTransformer(
            vocab_size=vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256
        )

        dataset = TextDataset(data, seq_len=32)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        return model, dataloader, optimizer, criterion, tokenizer

    def test_loss_decreases(self, setup_training):
        """训练后 loss 应该下降"""
        model, dataloader, optimizer, criterion, _ = setup_training

        # 记录初始 loss
        model.train()
        initial_loss = None
        for _ in range(5):  # 训练 5 个 epoch
            epoch_loss = train(model, dataloader, optimizer, criterion, torch.device('cpu'), 0)
            if initial_loss is None:
                initial_loss = epoch_loss

        # 最终 loss 应该低于初始 loss
        final_loss = epoch_loss
        assert final_loss < initial_loss, \
            f"Loss 未下降: 初始 {initial_loss:.4f}, 最终 {final_loss:.4f}"

    def test_loss_reduction_threshold(self, setup_training):
        """训练 loss 下降 >= 30%"""
        model, dataloader, optimizer, criterion, _ = setup_training

        model.train()
        initial_loss = None
        best_loss = float('inf')

        for epoch in range(50):  # 训练 50 个 epoch
            epoch_loss = train(model, dataloader, optimizer, criterion, torch.device('cpu'), epoch)
            if initial_loss is None:
                initial_loss = epoch_loss
            best_loss = min(best_loss, epoch_loss)

        reduction = (initial_loss - best_loss) / initial_loss * 100
        assert reduction >= 30, \
            f"Loss 下降不足 30%: 实际下降 {reduction:.1f}%"

    def test_gradients_update(self, setup_training):
        """参数在训练中被更新"""
        model, dataloader, optimizer, criterion, _ = setup_training

        # 记录初始参数
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # 训练一步
        model.train()
        train(model, dataloader, optimizer, criterion, torch.device('cpu'), 0)

        # 检查参数是否变化
        for name, param in model.named_parameters():
            assert not torch.equal(initial_params[name], param), \
                f"参数 {name} 未被更新"


class TestGenerationPipeline:
    """生成流水线集成测试"""

    @pytest.fixture
    def trained_model(self):
        """训练一个简单模型用于生成测试"""
        text, tokenizer, data = get_dataset('shakespeare')
        vocab_size = tokenizer.vocab_size

        model = MiniTransformer(
            vocab_size=vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256
        )

        # 快速训练
        dataset = TextDataset(data, seq_len=32)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(20):
            train(model, dataloader, optimizer, criterion, torch.device('cpu'), 0)

        return model, tokenizer

    def test_generation_produces_tokens(self, trained_model):
        """生成产生指定数量的 token"""
        model, tokenizer = trained_model

        generated = generate(
            model, tokenizer, "To be",
            max_new_tokens=20,
            temperature=0.8,
            device='cpu'
        )

        new_tokens = len(generated) - len("To be")
        assert new_tokens >= 20, f"生成 token 数不足: 期望 >=20, 实际 {new_tokens}"

    def test_generation_starts_with_prompt(self, trained_model):
        """生成结果以 prompt 开头"""
        model, tokenizer = trained_model
        prompt = "To be"

        generated = generate(
            model, tokenizer, prompt,
            max_new_tokens=10,
            device='cpu'
        )

        assert generated.startswith(prompt), \
            f"生成结果不以 prompt 开头: '{generated}'"

    def test_temperature_effect(self, trained_model):
        """温度参数影响生成多样性"""
        model, tokenizer = trained_model
        prompt = "To be"

        # 低温应该更确定
        results_low_temp = set()
        for _ in range(5):
            gen = generate(model, tokenizer, prompt, max_new_tokens=5, temperature=0.1, device='cpu')
            results_low_temp.add(gen)

        # 高温应该更多样
        results_high_temp = set()
        for _ in range(5):
            gen = generate(model, tokenizer, prompt, max_new_tokens=5, temperature=2.0, device='cpu')
            results_high_temp.add(gen)

        # 高温生成的多样性应该更高或相等
        assert len(results_high_temp) >= len(results_low_temp), \
            "高温生成的多样性应该更高"


class TestEndToEnd:
    """端到端测试"""

    def test_train_save_load_generate(self):
        """完整流程: 训练 → 保存 → 加载 → 生成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 训练
            text, tokenizer, data = get_dataset('shakespeare')
            vocab_size = tokenizer.vocab_size

            model = MiniTransformer(
                vocab_size=vocab_size,
                d_model=64,
                num_heads=4,
                num_layers=2,
                d_ff=256
            )

            dataset = TextDataset(data, seq_len=32)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for _ in range(20):
                train(model, dataloader, optimizer, criterion, torch.device('cpu'), 0)

            # 2. 保存
            save_path = os.path.join(tmpdir, 'test_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': vocab_size,
                    'd_model': 64,
                    'num_heads': 4,
                    'num_layers': 2,
                    'd_ff': 256,
                    'max_len': 128,
                },
                'tokenizer_chars': tokenizer.chars,
            }, save_path)

            # 3. 加载
            checkpoint = torch.load(save_path)
            loaded_model = MiniTransformer(
                vocab_size=checkpoint['config']['vocab_size'],
                d_model=checkpoint['config']['d_model'],
                num_heads=checkpoint['config']['num_heads'],
                num_layers=checkpoint['config']['num_layers'],
                d_ff=checkpoint['config']['d_ff'],
            )
            loaded_model.load_state_dict(checkpoint['model_state_dict'])

            # 4. 生成
            loaded_tokenizer = CharTokenizer('')
            loaded_tokenizer.chars = checkpoint['tokenizer_chars']
            loaded_tokenizer.vocab_size = len(loaded_tokenizer.chars)
            loaded_tokenizer.char_to_idx = {c: i for i, c in enumerate(loaded_tokenizer.chars)}
            loaded_tokenizer.idx_to_char = {i: c for i, c in enumerate(loaded_tokenizer.chars)}

            generated = generate(
                loaded_model, loaded_tokenizer, "To be",
                max_new_tokens=20,
                device='cpu'
            )

            assert len(generated) > len("To be"), "生成失败"
            assert generated.startswith("To be"), "生成不以 prompt 开头"


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
