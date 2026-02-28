"""
Mini Transformer - 推理脚本

自回归生成文本
目标：生成 ≥20 个 token
"""

import torch
from transformer import MiniTransformer
from data import CharTokenizer


def generate(
    model: MiniTransformer,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    device: str = 'cpu'
) -> str:
    """
    自回归生成文本

    Args:
        model: 训练好的模型
        tokenizer: tokenizer
        prompt: 输入提示
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数（越高越随机）
        top_k: Top-k 采样（None 表示不使用）
        device: 设备

    Returns:
        生成的文本
    """
    model.eval()

    # 编码 prompt
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 获取最近的 seq_len 个 token
            idx_cond = ids[:, -model.d_model * 2:]  # 限制长度

            # 前向传播
            logits = model(idx_cond)

            # 只取最后一个位置的 logits
            logits = logits[:, -1, :] / temperature

            # Top-k 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Softmax 得到概率
            probs = torch.softmax(logits, dim=-1)

            # 采样
            next_id = torch.multinomial(probs, num_samples=1)

            # 拼接
            ids = torch.cat([ids, next_id], dim=1)

    # 解码
    generated = tokenizer.decode(ids[0].tolist())
    return generated


def main():
    print("=" * 50)
    print("Mini Transformer 推理")
    print("=" * 50)

    # 检查是否有保存的模型
    import os
    if not os.path.exists('final_model.pt'):
        print("\n错误: 未找到训练好的模型 (final_model.pt)")
        print("请先运行 train.py 训练模型")
        return

    # 加载模型
    checkpoint = torch.load('final_model.pt', map_location='cpu')
    config = checkpoint['config']

    # 重建 tokenizer
    tokenizer = CharTokenizer('')
    tokenizer.chars = checkpoint['tokenizer_chars']
    tokenizer.vocab_size = len(tokenizer.chars)
    tokenizer.char_to_idx = {c: i for i, c in enumerate(tokenizer.chars)}
    tokenizer.idx_to_char = {i: c for i, c in enumerate(tokenizer.chars)}

    # 创建模型
    model = MiniTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=0.0  # 推理时不用 dropout
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    device = config['device']
    model.to(device)

    print(f"\n模型加载成功!")
    print(f"词表大小: {tokenizer.vocab_size}")

    # 测试生成
    prompts = [
        "To be",
        "Whether",
        "The ",
    ]

    print("\n" + "-" * 50)
    print("生成测试:")
    print("-" * 50)

    total_tokens = 0
    for prompt in prompts:
        generated = generate(
            model, tokenizer, prompt,
            max_new_tokens=50,
            temperature=0.8,
            device=device
        )
        new_tokens = len(generated) - len(prompt)
        total_tokens = max(total_tokens, new_tokens)

        print(f"\nPrompt: '{prompt}'")
        print(f"生成 ({new_tokens} 新 tokens):")
        print(f"  {generated}")

    # 验收标准
    print("\n" + "=" * 50)
    if total_tokens >= 20:
        print(f"✓ 验收标准达成: 生成 token 数 ≥20 (实际: {total_tokens})")
    else:
        print(f"✗ 验收标准未达成: 生成 token 数 <20 (实际: {total_tokens})")


if __name__ == "__main__":
    main()
