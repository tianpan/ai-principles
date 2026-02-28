## Context

现有项目已完成基础功能，但用户反馈可视化效果不够酷炫，缺乏 3Blue1Brown 风格的动画展示。需要全面升级视觉设计和交互体验。

## Goals / Non-Goals

**Goals:**
- 首页大屏震撼的 3Blue1Brown 风格动画
- 交互实验室支持 Pyodide 运行 Mini Transformer
- 全站组件升级为动态动画版本
- 统一的深空科技风视觉设计

**Non-Goals:**
- 不改变现有部署方式（仍为 GitHub Pages）
- 不增加后端 API（纯前端 + Pyodide）
- 不支持 IE 浏览器

## Decisions

### 1. 动画技术栈：D3.js + Framer Motion + Canvas

**选择理由：**
- D3.js 是数据驱动可视化的标准，适合 Attention 矩阵等
- Framer Motion 提供流畅的 React 动画
- Canvas 用于粒子背景和复杂动画场景

### 2. Mini Transformer 运行：Pyodide + Web Worker

**选择理由：**
- 用户要求在浏览器内运行，不需要离开网站
- Web Worker 确保不阻塞 UI
- 预加载策略减少等待时间

### 3. 首页动画设计：30秒循环

**Phase 1 (0-5s):** 开场 - 粒子汇聚 + 标题淡入
**Phase 2 (5-15s):** Token 流动 - 字符 → 向量 → 连线
**Phase 3 (15-25s):** Attention 可视化 - 热力图填充 + 连线动画
**Phase 4 (25-30s):** 输出生成 - 新 Token 预测 → 循环

### 4. 组件升级策略

| 组件 | 当前 | 升级后 |
|------|------|--------|
| AttentionExplorer | 静态热力图 | 动态热力图 + 连线 + Tooltip |
| QKVStepByStep | 分步展示 | 流畅向量变换动画 |
| MultiHeadComparator | 静态对比 | 动态切换 + 过渡动画 |
| PositionalEncoding | 静态波形 | 动态波形动画 |

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| Pyodide 加载慢（~10MB） | 首页预加载 + 进度提示 + 降级方案（预计算模式） |
| 动画性能问题 | CSS 动画优先，Canvas 仅用于复杂场景 |
| 开发周期长 | 分阶段交付，优先首页和实验室 |

## File Structure

```
src/
├── components/
│   ├── hero/                    # 首页大屏
│   │   ├── HeroAnimation.tsx
│   │   ├── ParticleBg.tsx
│   │   ├── TokenFlow.tsx
│   │   └── AttentionViz.tsx
│   │
│   ├── lab/                     # 交互实验室
│   │   ├── LabPage.tsx
│   │   ├── PyodideRunner.tsx
│   │   ├── TrainingPanel.tsx
│   │   └── InferencePanel.tsx
│   │
│   ├── visualizations/          # 升级现有组件
│   │   ├── AttentionExplorer.tsx
│   │   ├── QKVStepByStep.tsx
│   │   ├── MultiHeadComparator.tsx
│   │   ├── PositionalEncoding.tsx
│   │   └── DataFlowAnimation.tsx
│   │
│   └── common/
│       ├── AnimatedCard.tsx
│       ├── GlowingButton.tsx
│       └── LoadingSpinner.tsx
│
├── hooks/
│   ├── usePyodide.ts
│   ├── useAnimation.ts
│   └── useTransformer.ts
│
└── workers/
    └── transformer.worker.ts
```
