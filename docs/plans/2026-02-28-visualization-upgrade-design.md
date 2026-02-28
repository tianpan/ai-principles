# Transformer 可视化平台升级设计

> **版本**: v1.0
> **创建日期**: 2026-02-28
> **状态**: 设计确认

---

## 1. 背景

现有项目已部署到 GitHub Pages，但缺乏真正酷炫的可视化效果。用户期望：
- 3Blue1Brown 风格的动画展示
- 深空科技风视觉设计
- 真正的交互式体验
- Mini Transformer 在浏览器内运行

## 2. 目标

### 2.1 首页大屏
- 震撼的 3Blue1Brown 风格开场动画
- Token → Embedding → Attention → Output 完整流程可视化
- 动态热力图 + 粒子背景

### 2.2 交互实验室
- 用户输入文本，实时计算 Attention
- 可视化 Q、K、V 矩阵变换过程
- Pyodide 在浏览器内运行 Mini Transformer

### 2.3 全站组件升级
- 所有可视化组件升级为动画版本
- 统一的深空科技风设计语言
- 响应式适配

## 3. 技术方案

### 3.1 动画技术栈
- **D3.js**: 数据驱动可视化
- **Framer Motion**: React 动画库
- **Canvas API**: 粒子背景、复杂动画
- **CSS Animations**: 基础动画效果

### 3.2 Mini Transformer 运行
- **Pyodide**: 在浏览器中运行 Python
- **Web Worker**: 后台运行模型，不阻塞 UI
- **预加载策略**: 首页加载时开始下载 Pyodide

### 3.3 视觉规范
```
主背景: #0a0f1a (深空黑蓝)
次背景: #111827, #1a2234
主强调: #06b6d4 (青色)
辅助色: #10b981 (绿色), #f59e0b (琥珀)
文字色: #f1f5f9 (主), #94a3b8 (次), #64748b (弱)
```

## 4. 详细设计

### 4.1 首页动画序列（30秒循环）

**Phase 1 (0-5s): 开场**
- 深空背景渐入
- 粒子从四周汇聚到中心
- 标题 "Attention Is All You Need" 淡入

**Phase 2 (5-15s): Token 流动**
- "Hello World" 字符逐个出现
- 每个字符 → 发光球体 → 向量条形图
- 向量之间产生连线动画

**Phase 3 (15-25s): Attention 可视化**
- 热力图从左上角开始填充
- 颜色从暗到亮渐变
- 连线从 Token 指向热力图
- 高亮单元格时显示权重值

**Phase 4 (25-30s): 输出生成**
- 热力图 → 新 Token 预测
- 新 Token 动画出现
- 循环回 Phase 2

### 4.2 交互实验室布局

```
┌─────────────────────────────────────────────────────────────────┐
│ 实验室控制面板                                                   │
├─────────────────────────────────────────────────────────────────┤
│ 模式选择: [训练模式] [推理模式] [探索模式]                        │
│ 模型参数: Heads: [4▼] Layers: [2▼] d_model: [64▼]              │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ 输入面板         │ │ 可视化面板       │ │ 输出面板         │
│                  │ │                  │ │                  │
│ [文本输入框]     │ │ [Attention热力图]│ │ [生成文本]       │
│                  │ │                  │ │                  │
│ 示例:            │ │ [Q/K/V矩阵]      │ │ 概率分布:        │
│ • To be, or...   │ │                  │ │ ▓▓▓▓▓░░░░ 55%    │
│ • Hello world    │ │ [数据流动画]      │ │ ▓▓▓░░░░░░ 33%    │
│ • The quick...   │ │                  │ │ ▓░░░░░░░░ 12%    │
│                  │ │ [训练曲线]       │ │                  │
│ [运行] [重置]    │ │                  │ │ [复制] [保存]    │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

### 4.3 组件升级清单

| 组件 | 当前状态 | 升级目标 |
|------|---------|---------|
| AttentionExplorer | 静态热力图 | 动画热力图 + 连线动画 + Tooltip |
| QKVStepByStep | 分步展示 | 流畅向量变换动画 |
| MultiHeadComparator | 静态对比 | 动态切换 + 过渡动画 |
| PositionalEncoding | 静态波形 | 动态波形动画 |
| MiniTransformerLab | 独立脚本 | 网页内 Pyodide 运行 |

## 5. 文件结构

```
src/
├── components/
│   ├── hero/                    # 首页大屏
│   │   ├── HeroAnimation.tsx    # 主动画组件
│   │   ├── ParticleBg.tsx       # 粒子背景
│   │   ├── TokenFlow.tsx        # Token 流动
│   │   └── AttentionViz.tsx     # Attention 可视化
│   │
│   ├── lab/                     # 交互实验室
│   │   ├── LabPage.tsx          # 实验室主页面
│   │   ├── PyodideRunner.tsx    # Pyodide 运行器
│   │   ├── TrainingPanel.tsx    # 训练控制面板
│   │   └── InferencePanel.tsx   # 推理控制面板
│   │
│   ├── visualizations/          # 可视化组件（升级）
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
├── workers/
│   └── transformer.worker.ts
│
└── styles/
    ├── animations.css
    └── components.css
```

## 6. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Pyodide 加载慢（~10MB） | 首页预加载 + 进度提示 + 降级方案 |
| 动画性能问题 | 使用 CSS 动画优先，Canvas 用于复杂场景 |
| 开发周期长 | 分阶段交付，优先首页和实验室 |

## 7. 验收标准

1. **首页大屏**: 动画流畅运行，30秒循环无卡顿
2. **交互实验室**: 用户输入文本后 5 秒内显示结果
3. **视觉效果**: 符合 3Blue1Brown 风格，深空科技感
4. **响应式**: 桌面端和移动端均可用
