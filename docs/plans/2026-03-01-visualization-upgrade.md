# Transformer 可视化升级实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建 3Blue1Brown 风格的酷炫 Transformer 可视化平台，包含首页动画、Pyodide 实验室和升级的交互组件。

**Architecture:** 首页使用 Canvas + D3.js 实现 3Blue1Brown 风格动画；实验室使用 Pyodide 在浏览器中运行 Mini Transformer；所有可视化组件升级为动态动画版本。

**Tech Stack:** React, TypeScript, Framer Motion, D3.js, Canvas API, Pyodide, Tailwind CSS

---

## Phase 1: 依赖安装与基础设施

### Task 1.1: 安装 D3.js

**Files:**
- Modify: `package.json`

**Step 1: 安装 D3.js**

```bash
cd /Users/admin/Documents/00-Projects/2026-02-27-AI原理科普
npm install d3 @types/d3
```

**Step 2: 验证安装**

Run: `npm list d3`
Expected: `d3@7.x.x`

---

### Task 1.2: 创建动画工具类和 CSS 变量

**Files:**
- Create: `src/styles/animations.css`
- Modify: `src/styles/global.css`

**Step 1: 创建 animations.css**

```css
/* src/styles/animations.css */

/* 入场动画 */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes glowPulse {
  0%, 100% {
    box-shadow: 0 0 5px rgba(6, 182, 212, 0.3);
  }
  50% {
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.6);
  }
}

@keyframes particleFloat {
  0%, 100% {
    transform: translateY(0) translateX(0);
  }
  25% {
    transform: translateY(-10px) translateX(5px);
  }
  50% {
    transform: translateY(-5px) translateX(-5px);
  }
  75% {
    transform: translateY(-15px) translateX(3px);
  }
}

/* 热力图单元格动画 */
@keyframes cellFadeIn {
  from {
    opacity: 0;
    transform: scale(0.8);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

/* 数据流动画 */
@keyframes dataFlow {
  0% {
    stroke-dashoffset: 100;
    opacity: 0;
  }
  50% {
    opacity: 1;
  }
  100% {
    stroke-dashoffset: 0;
    opacity: 1;
  }
}

/* 工具类 */
.animate-fade-in-up {
  animation: fadeInUp 0.5s ease-out forwards;
}

.animate-glow-pulse {
  animation: glowPulse 2s ease-in-out infinite;
}

.animate-particle-float {
  animation: particleFloat 3s ease-in-out infinite;
}

.animate-cell-fade-in {
  animation: cellFadeIn 0.3s ease-out forwards;
}

.animate-data-flow {
  animation: dataFlow 1s ease-out forwards;
}

/* 延迟工具类 */
.delay-100 { animation-delay: 100ms; }
.delay-200 { animation-delay: 200ms; }
.delay-300 { animation-delay: 300ms; }
.delay-400 { animation-delay: 400ms; }
.delay-500 { animation-delay: 500ms; }

/* 减少动画偏好 */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

**Step 2: 在 global.css 中导入**

在 `src/styles/global.css` 末尾添加：

```css
/* 导入动画样式 */
@import './animations.css';
```

---

## Phase 2: 首页大屏动画

### Task 2.1: 创建粒子背景组件

**Files:**
- Create: `src/components/hero/ParticleBg.tsx`

```tsx
// src/components/hero/ParticleBg.tsx
import { useEffect, useRef } from 'react';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
}

export default function ParticleBg() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 设置画布大小
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // 初始化粒子
    const particleCount = 150;
    particlesRef.current = Array.from({ length: particleCount }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 2 + 1,
      opacity: Math.random() * 0.5 + 0.2,
    }));

    // 动画循环
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 绘制粒子
      particlesRef.current.forEach((p) => {
        // 更新位置
        p.x += p.vx;
        p.y += p.vy;

        // 边界检查
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        // 绘制
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(6, 182, 212, ${p.opacity})`;
        ctx.fill();
      });

      // 绘制连线
      particlesRef.current.forEach((p1, i) => {
        particlesRef.current.slice(i + 1).forEach((p2) => {
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 100) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(6, 182, 212, ${0.1 * (1 - dist / 100)})`;
            ctx.stroke();
          }
        });
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: -1 }}
    />
  );
}
```

---

### Task 2.2: 创建 Token 流动动画组件

**Files:**
- Create: `src/components/hero/TokenFlow.tsx`

```tsx
// src/components/hero/TokenFlow.tsx
import { motion } from 'framer-motion';

const tokens = ['T', 'r', 'a', 'n', 's', 'f', 'o', 'r', 'm', 'e', 'r'];

export default function TokenFlow() {
  return (
    <div className="flex items-center justify-center gap-2 my-8">
      {tokens.map((token, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, scale: 0, y: -50 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{
            duration: 0.5,
            delay: i * 0.1,
            repeat: Infinity,
            repeatDelay: 3,
          }}
          className="relative"
        >
          {/* Token 方块 */}
          <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-cyan-500/30">
            {token}
          </div>

          {/* 发光效果 */}
          <motion.div
            className="absolute inset-0 rounded-lg bg-cyan-400 blur-xl opacity-30"
            animate={{
              opacity: [0.2, 0.5, 0.2],
              scale: [1, 1.2, 1],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: i * 0.1,
            }}
          />

          {/* 连接线 */}
          {i < tokens.length - 1 && (
            <motion.div
              className="absolute top-1/2 left-full w-2 h-0.5 bg-cyan-400"
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{
                duration: 0.3,
                delay: i * 0.1 + 0.3,
                repeat: Infinity,
                repeatDelay: 3,
              }}
              style={{ transformOrigin: 'left' }}
            />
          )}
        </motion.div>
      ))}
    </div>
  );
}
```

---

### Task 2.3: 创建 Attention 可视化组件

**Files:**
- Create: `src/components/hero/AttentionViz.tsx`

```tsx
// src/components/hero/AttentionViz.tsx
import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';

const tokens = ['The', 'cat', 'sat', 'on', 'mat'];

// 模拟注意力权重
const generateAttention = () => {
  return tokens.map((_, i) =>
    tokens.map((_, j) => {
      if (j > i) return 0; // 因果掩码
      const dist = i - j;
      return Math.exp(-dist * 0.3) + Math.random() * 0.2;
    }).map((v, _, arr) => v / arr.reduce((a, b) => a + b, 0))
  );
};

export default function AttentionViz() {
  const [attention, setAttention] = useState<number[][]>([]);

  useEffect(() => {
    setAttention(generateAttention());
    const interval = setInterval(() => {
      setAttention(generateAttention());
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const getColor = (weight: number) => {
    if (weight === 0) return 'bg-slate-800';
    const intensity = Math.min(weight * 4, 1);
    return `rgba(6, 182, 212, ${intensity})`;
  };

  return (
    <div className="my-8">
      <h3 className="text-center text-cyan-400 font-bold mb-4">Attention 热力图</h3>

      <div className="flex justify-center">
        <div>
          {/* 列标题 */}
          <div className="flex mb-2">
            <div className="w-12 h-8" />
            {tokens.map((t, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.1 }}
                className="w-16 h-8 flex items-center justify-center text-slate-400 text-sm"
              >
                {t}
              </motion.div>
            ))}
          </div>

          {/* 热力图 */}
          {attention.map((row, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex"
            >
              <div className="w-12 h-12 flex items-center justify-center text-slate-400 text-sm">
                {tokens[i]}
              </div>
              {row.map((w, j) => (
                <motion.div
                  key={j}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: i * 0.1 + j * 0.05 }}
                  className="w-16 h-12 flex items-center justify-center text-xs font-mono rounded m-0.5"
                  style={{ backgroundColor: getColor(w) }}
                >
                  {w > 0 ? w.toFixed(2) : '-'}
                </motion.div>
              ))}
            </motion.div>
          ))}
        </div>
      </div>

      <p className="text-center text-slate-500 text-sm mt-4">
        每个位置只能看到自己和之前的内容（因果掩码）
      </p>
    </div>
  );
}
```

---

### Task 2.4: 创建首页主组件

**Files:**
- Create: `src/components/hero/HeroAnimation.tsx`

```tsx
// src/components/hero/HeroAnimation.tsx
import { motion } from 'framer-motion';
import ParticleBg from './ParticleBg';
import TokenFlow from './TokenFlow';
import AttentionViz from './AttentionViz';

export default function HeroAnimation() {
  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
      {/* 粒子背景 */}
      <ParticleBg />

      {/* 主内容 */}
      <div className="relative z-10 text-center px-4">
        {/* 标题 */}
        <motion.h1
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-5xl md:text-7xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent"
        >
          Attention Is All You Need
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="text-xl text-slate-400 mb-8"
        >
          从零开始理解 Transformer
        </motion.p>

        {/* Token 流动 */}
        <TokenFlow />

        {/* Attention 可视化 */}
        <div className="max-w-2xl mx-auto">
          <AttentionViz />
        </div>

        {/* CTA 按钮 */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5, duration: 0.5 }}
          className="flex gap-4 justify-center mt-8"
        >
          <a
            href="/prerequisites"
            className="px-8 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-medium hover:shadow-lg hover:shadow-cyan-500/30 transition-all"
          >
            开始学习
          </a>
          <a
            href="/lab/intro"
            className="px-8 py-3 rounded-lg border border-cyan-500 text-cyan-400 font-medium hover:bg-cyan-500/10 transition-all"
          >
            进入实验室
          </a>
        </motion.div>
      </div>
    </div>
  );
}
```

---

### Task 2.5: 更新首页内容

**Files:**
- Modify: `src/content/docs/index.mdx`

在文件开头添加：

```mdx
---
title: 欢迎
description: 从零开始理解 AI，亲手实现 Transformer
---

import HeroAnimation from '../../components/hero/HeroAnimation.astro';

<HeroAnimation />

---

## 为什么选择这个教程？
...
```

---

## Phase 3: 交互组件升级

### Task 3.1: 升级 AttentionExplorer 组件

**Files:**
- Modify: `src/components/visualizations/AttentionExplorer.tsx`

添加动画效果和连线。

---

## Phase 4: 验收与部署

### Task 4.1: 本地测试

```bash
npm run dev
```

验收：
- [ ] 首页动画流畅运行
- [ ] 粒子背景正常显示
- [ ] Token 流动动画正常
- [ ] Attention 热力图动画正常

### Task 4.2: 构建与部署

```bash
npm run build
git add .
git commit -m "feat: 添加 3Blue1Brown 风格首页动画"
git push
```
