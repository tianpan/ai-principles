## Context

全新项目，目标是在 5 周内交付一个可公开访问的 Transformer 科普产品。核心约束：
- 零基础可达性优先
- 可验证学习结果（用户能从零实现最小 Transformer）
- 桌面端与移动端可用

## Goals / Non-Goals

**Goals:**
- 搭建 Astro + Starlight 内容骨架，支持 React 交互组件
- 完成 Chapter 2（Transformer 核心机制）完整闭环
- 5 个核心交互模块可运行
- Mini Transformer Lab 可训练、可推理
- 满足 4 项验收标准

**Non-Goals:**
- 不做完整课程平台（考试、社交、打卡）
- 不做 CNN/RNN/LSTM 等早期架构
- 不做后端 API（首版本地静态数据）
- 不追求榜单追新

## Decisions

### 1. 框架选择：Astro + Starlight

**选择理由：**
- 内容优先架构，MDX 支持好
- Islands 架构，交互组件按需加载，性能优
- 内置文档导航、搜索、主题切换

**替代方案：**
- Next.js：过于重量级，SEO 配置复杂
- VitePress：交互组件集成不如 Astro 灵活

### 2. 可视化：D3.js + Motion for React

**选择理由：**
- D3.js 是数据驱动可视化的标准
- Motion（原 Framer Motion）React 生态原生支持

**替代方案：**
- Three.js：3D 效果好但复杂度高，MVP 不需要
- 纯 CSS 动画：无法实现数据驱动的复杂可视化

### 3. 样式：Tailwind CSS + CSS Variables (Design Tokens)

**选择理由：**
- Tailwind 开发效率高
- CSS Variables 支持主题切换和 Design Tokens

### 4. 交互组件架构

```
src/components/
├── visualizations/           # D3.js 可视化组件
│   ├── AttentionExplorer.tsx
│   ├── QKVStepByStep.tsx
│   ├── MultiHeadComparator.tsx
│   └── PositionalEncodingToggle.tsx
├── lab/                      # 实验室组件
│   └── MiniTransformerLab.tsx
└── common/                   # 通用组件
    ├── TerminologyCard.tsx
    └── SummaryBox.tsx
```

### 5. Mini Transformer Lab 技术方案

**模型架构：**
- Embedding: vocab_size × d_model
- Positional Encoding: sin/cos 固定编码
- Self-Attention: 单头 → 多头扩展
- FFN: d_model → d_ff → d_model
- 残差连接 + LayerNorm

**训练方案：**
- 数据集：toy 文本（如莎士比亚片段、简单对话）
- 优化器：AdamW
- 损失函数：CrossEntropyLoss
- 目标：loss 下降 ≥30%

**推理方案：**
- 自回归生成
- 支持 Greedy / Top-k / Top-p 解码

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| 交互炫但学习路径混乱 | 每个交互统一"现象-解释-代码映射"结构 |
| 范围回弹导致失焦 | 严格锁定 Chapter 2 MVP，后续迭代扩展 |
| 实现难度高导致延期 | 先保底最小模型，再迭代高级功能 |
| 移动端交互体验差 | 响应式设计优先，复杂交互桌面端优化 |

## Open Questions

1. Mini Transformer Lab 是嵌入 Web（Pyodide）还是独立 Python 脚本？
   - **建议**：MVP 阶段独立 Python 脚本 + Colab 运行，后续迭代 WebAssembly

2. 部署平台：Vercel vs GitHub Pages？
   - **建议**：GitHub Pages 优先（免费、稳定），Vercel 作为备选
