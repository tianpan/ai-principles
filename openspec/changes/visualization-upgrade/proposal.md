## Why

现有项目已部署到 GitHub Pages，但缺乏真正酷炫的可视化效果。用户期望看到 3Blue1Brown 风格的动画展示、深空科技风视觉设计、真正的交互式体验，以及 Mini Transformer 在浏览器内运行。当前实现只是基础的静态组件 + 命令行 Python 脚本，无法达到震撼的展示效果。

## What Changes

**升级**
- 首页大屏：3Blue1Brown 风格开场动画（Token 流动 → Attention 热力图 → 输出生成）
- 交互实验室：完整的 Pyodide 运行环境，用户输入文本实时看到 Transformer 运行结果
- 全站可视化组件：从静态升级为动态动画版本
- 视觉设计：统一的深空科技风，粒子背景、发光效果、流畅过渡

**验收标准**
1. 首页动画流畅运行，30秒循环无卡顿
2. 用户输入文本后 5 秒内显示 Attention 计算结果
3. 视觉效果符合 3Blue1Brown 风格，深空科技感
4. 响应式适配：桌面端和移动端均可用

## Capabilities

### New Capabilities

- `hero-animation`: 首页大屏 3Blue1Brown 风格动画（粒子背景 + Token 流动 + Attention 可视化）
- `pyodide-lab`: Pyodide 运行环境 + Web Worker 后台计算
- `interactive-lab`: 完整的交互实验室（训练/推理/探索三种模式）

### Modified Capabilities

- `attention-visualizer`: 升级为动态热力图 + 连线动画 + Tooltip
- `multihead-visualizer`: 升级为动态切换 + 过渡动画
- `positional-encoding`: 升级为动态波形动画
- `mini-transformer-lab`: 从独立脚本升级为网页内 Pyodide 运行

## Impact

- 现有组件需要重构，但保持 API 兼容
- 新增 Pyodide 依赖（~10MB），需要预加载策略
- 首页完全重新设计
- 部署方式不变，仍为 GitHub Pages
