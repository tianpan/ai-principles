# Hero Animation Spec

## 功能描述

首页大屏展示 3Blue1Brown 风格的 Transformer 动画，30秒循环，包含：
1. 粒子背景动画
2. Token 流动动画
3. Attention 热力图动画
4. 输出生成动画

## 技术要求

### 粒子背景 (ParticleBg.tsx)
- 使用 Canvas API
- 100-200 个粒子
- 粒子从四周汇聚到中心
- 颜色：青色 (#06b6d4) 带发光效果

### Token 流动 (TokenFlow.tsx)
- 使用 D3.js + SVG
- 动画序列：
  1. 字符逐个出现（间隔 200ms）
  2. 字符 → 发光球体
  3. 球体 → 向量条形图
  4. 向量之间产生连线

### Attention 可视化 (AttentionViz.tsx)
- 使用 CSS Grid + Framer Motion
- 热力图从左上角开始填充
- 颜色渐变：暗 → 亮
- 连线动画：Token → 热力图
- 高亮单元格显示权重值

## 验收标准

- [ ] 动画流畅运行，FPS ≥ 30
- [ ] 30秒循环无卡顿
- [ ] 移动端适配（简化动画）
- [ ] 支持 prefers-reduced-motion
