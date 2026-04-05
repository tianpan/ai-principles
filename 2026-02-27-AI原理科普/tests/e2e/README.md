# 测试说明

## 测试结构

```
tests/
├── e2e/                      # E2E 测试 (Playwright)
│   ├── app.spec.ts           # Web 应用测试
│   └── playwright.config.ts  # Playwright 配置
│
mini_transformer/
└── tests/
    ├── test_units.py         # 单元测试
    └── test_integration.py   # 集成测试
```

## 运行测试

### 单元测试 (Python)

```bash
cd mini_transformer
pip install pytest torch

# 运行所有单元测试
pytest tests/test_units.py -v

# 运行带覆盖率
pytest tests/test_units.py --cov=. --cov-report=html
```

### 集成测试 (Python)

```bash
cd mini_transformer

# 运行集成测试
pytest tests/test_integration.py -v

# 运行所有测试
pytest tests/ -v
```

### E2E 测试 (Playwright)

```bash
# 安装依赖
npm install -D @playwright/test
npx playwright install

# 启动开发服务器（新终端）
npm run dev

# 运行 E2E 测试
npx playwright test

# 带 UI 运行
npx playwright test --ui

# 生成报告
npx playwright show-report
```

## 测试覆盖

### 单元测试 (test_units.py)
- ✅ TokenEmbedding: 维度、缩放因子
- ✅ PositionalEncoding: sin/cos、位置唯一性
- ✅ SelfAttention: Q/K/V、注意力权重
- ✅ MultiHeadAttention: 头分割、输出维度
- ✅ TransformerBlock: 残差连接
- ✅ MiniTransformer: 完整前向传播、梯度流

### 集成测试 (test_integration.py)
- ✅ CharTokenizer: 编码解码一致性
- ✅ 数据集: 批次生成、next-token 预测
- ✅ 训练流程: loss 下降、参数更新
- ✅ 生成流程: token 数量、温度效果
- ✅ 端到端: 训练→保存→加载→生成

### E2E 测试 (app.spec.ts)
- ✅ 页面可访问性
- ✅ 导航功能
- ✅ 组件交互
- ✅ 响应式设计
- ✅ 性能检查
- ✅ 可访问性

## 验收标准

| 测试类型 | 目标 | 验证 |
|---------|------|------|
| 单元测试 | 100% 通过 | `pytest tests/test_units.py` |
| 集成测试 | Loss 下降 ≥30%, 生成 ≥20 tokens | `pytest tests/test_integration.py` |
| E2E 测试 | 所有页面可访问 | `npx playwright test` |
