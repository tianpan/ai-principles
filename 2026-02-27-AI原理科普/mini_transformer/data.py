"""
Mini Transformer - 数据准备

Toy 数据集：莎士比亚片段
"""

# 简化的莎士比亚文本片段
SHAKESPEARE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
"""

# 简单对话数据
DIALOGUE_TEXT = """
A: 你好，今天天气怎么样？
B: 今天天气很好，阳光明媚。
A: 我们去公园散步吧。
B: 好主意，我也想去公园。
A: 那我们什么时候出发？
B: 下午两点怎么样？
A: 可以，我两点在门口等你。
B: 好的，不见不散。
"""


class CharTokenizer:
    """字符级 Tokenizer"""

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list:
        """将文本编码为 token IDs"""
        return [self.char_to_idx[c] for c in text]

    def decode(self, ids: list) -> str:
        """将 token IDs 解码为文本"""
        return ''.join([self.idx_to_char[i] for i in ids])


def get_dataset(text_type: str = 'shakespeare'):
    """获取数据集和 tokenizer"""
    if text_type == 'shakespeare':
        text = SHAKESPEARE_TEXT
    else:
        text = DIALOGUE_TEXT

    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    return text, tokenizer, data


if __name__ == "__main__":
    text, tokenizer, data = get_dataset('shakespeare')

    print(f"文本长度: {len(text)}")
    print(f"词表大小: {tokenizer.vocab_size}")
    print(f"Token 数量: {len(data)}")

    # 测试编码解码
    test_text = "To be"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n测试编码: '{test_text}' -> {encoded}")
    print(f"测试解码: {encoded} -> '{decoded}'")
