## Why

零基础学习者缺乏一个能让其从"看懂Transformer"走到"从零实现最小Transformer"的交互式科普产品。现有资源要么过于理论，要么缺少可运行代码，无法形成完整学习闭环。

## What Changes

**新增**
- Astro + Starlight 项目骨架，支持 MDX 内容 + React 交互组件
- Chapter 2（Transformer 核心机制）完整内容：2.1 Token/Embedding → 2.6 FFN+输出层
- 5 个核心交互可视化模块
- Mini Transformer Lab（可运行训练与推理）
- 零基础导学路径（术语卡 + 预备知识）

**验收标准**
1. 代码实现：Embedding、位置编码、自注意力、多头、FFN、残差+LayerNorm
2. 训练可验证：toy 数据集训练后 loss 下降 ≥30%
3. 推理可运行：给定 prompt 生成 ≥20 token
4. 机制可解释：模块作用与代码位置一一对应

## Capabilities

### New Capabilities

- `astro-foundation`: Astro + Starlight + React + Tailwind 项目基础设施
- `chapter2-content`: Transformer 核心机制章节内容（2.1-2.6）
- `attention-visualizer`: Self-Attention 可视化交互组件（Attention Explorer + QKV Step-by-Step）
- `multihead-visualizer`: 多头注意力对比器
- `positional-encoding`: 位置编码可视化切换器
- `mini-transformer-lab`: 最小 Transformer 实验室（训练 + 推理）
- `learning-path`: 零基础导学路径与术语卡

### Modified Capabilities

无（全新项目）

## Impact

- 新项目，无现有代码影响
- 技术栈：Astro、Starlight、React、TypeScript、D3.js、Tailwind CSS
- 部署目标：Vercel / GitHub Pages
- 周期：5 周（MVP）→ 7 天执行计划
