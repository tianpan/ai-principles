"""
Mini Transformer - 训练脚本

使用 AdamW 优化器和 CrossEntropyLoss 训练模型
目标：loss 下降 ≥30%
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

from transformer import MiniTransformer
from data import get_dataset


class TextDataset(Dataset):
    """文本数据集"""

    def __init__(self, data: list, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        # 输入: data[idx:idx+seq_len]
        # 目标: data[idx+1:idx+seq_len+1] (下一个 token)
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # 前向传播
        logits = model(x)

        # 计算损失
        # logits: [batch, seq_len, vocab_size]
        # y: [batch, seq_len]
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """评估模型"""
    model.eval()
    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # 配置
    config = {
        'vocab_size': None,  # 由 tokenizer 决定
        'd_model': 64,
        'num_heads': 4,
        'num_layers': 2,
        'd_ff': 256,
        'max_len': 128,
        'dropout': 0.1,
        'seq_len': 32,
        'batch_size': 16,
        'learning_rate': 3e-4,
        'epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print("=" * 50)
    print("Mini Transformer 训练")
    print("=" * 50)

    # 准备数据
    text, tokenizer, data = get_dataset('shakespeare')
    config['vocab_size'] = tokenizer.vocab_size

    print(f"\n数据集信息:")
    print(f"  文本长度: {len(text)}")
    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  Token 数量: {len(data)}")

    # 创建数据集
    dataset = TextDataset(data, config['seq_len'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # 创建模型
    device = torch.device(config['device'])
    model = MiniTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # 训练
    print(f"\n开始训练 (设备: {config['device']})...")
    print("-" * 50)

    initial_loss = None
    best_loss = float('inf')

    for epoch in range(config['epochs']):
        start_time = time.time()
        loss = train(model, dataloader, optimizer, criterion, device, epoch)
        elapsed = time.time() - start_time

        if initial_loss is None:
            initial_loss = loss

        if loss < best_loss:
            best_loss = loss
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pt')

        # 每 10 个 epoch 打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            reduction = (initial_loss - loss) / initial_loss * 100
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                  f"下降: {reduction:.1f}% | 时间: {elapsed:.2f}s")

    # 最终结果
    print("-" * 50)
    final_reduction = (initial_loss - best_loss) / initial_loss * 100
    print(f"\n训练完成!")
    print(f"  初始 Loss: {initial_loss:.4f}")
    print(f"  最终 Loss: {best_loss:.4f}")
    print(f"  Loss 下降: {final_reduction:.1f}%")

    if final_reduction >= 30:
        print("  ✓ 验收标准达成: Loss 下降 ≥30%")
    else:
        print("  ✗ 验收标准未达成: Loss 下降 <30%")

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_chars': tokenizer.chars,
    }, 'final_model.pt')
    print(f"\n模型已保存到 final_model.pt")


if __name__ == "__main__":
    main()
