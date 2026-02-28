# AI原理科普项目 PRD

> **项目名称**：AI原理可视化科普系列
> **版本**：v2.1
> **创建日期**：2026-02-28
> **最后更新**：2026-02-28
> **作者**：Claude + 用户协作

---

## 一、项目背景与愿景

### 1.1 项目背景

生成式人工智能（GenAI）正在深刻改变世界，但其背后的原理——Transformer、预训练、对齐、Scaling Law 等概念——对于大多数人来说仍然抽象难懂。现有的科普内容要么过于浅显，要么过于学术，缺乏**直观、形象、系统**的解释。

> 当前范围约束：本项目暂不展开 CNN、RNN、LSTM、GRU 等早期架构专题，聚焦生成式 AI 主线。

### 1.2 项目愿景

打造一个**AI原理的可视化科普系列**，参考 Karpathy 的视频教程风格和 3Blue1Brown 的数学可视化方法，让复杂的AI概念变得**看得见、摸得着、理解得了**。

**风格定位**：酷炫、科技感、交互式

### 1.3 核心价值

| 价值点 | 描述 |
|--------|------|
| **降低门槛** | 让非技术背景的人也能理解AI原理 |
| **系统化** | 构建完整的知识体系，而非零散概念 |
| **可视化** | 用动画和交互替代枯燥的公式 |
| **启发式** | 不仅讲"是什么"，更讲"为什么" |

---

## 二、标杆项目研究

### 2.1 核心对标项目：Transformer Explainer

经过对 GitHub 和社区的充分调研，确定 **Transformer Explainer** 为核心对标项目。

| 维度 | 信息 |
|------|------|
| **GitHub** | https://github.com/poloclub/transformer-explainer |
| **Demo** | https://poloclub.github.io/transformer-explainer/ |
| **Stars** | ~6,800+ |
| **论文** | IEEE VIS 2024 |
| **许可** | MIT |

**核心特点**：
1. 在浏览器内运行实时 GPT-2 模型
2. 用户输入文本后可直接观察 Transformer 内部计算和预测变化
3. "概念解释 + 实时模型 + 参数调节"三合一
4. 明确提供 Live Demo、论文、视频三入口

**可借鉴的设计模式**：
- 实时交互 + 可视化反馈闭环
- 分层信息架构（概览 → 细节）
- 零门槛访问（无需安装、无需 GPU）

### 2.2 其他参考项目

| 项目 | Stars | 核心借鉴点 |
|------|-------|------------|
| **TensorFlow Playground** | ~12,800 | 极简控件、即时反馈、颜色语义一致 |
| **Diffusion Explainer** | ~450 | 多层抽象切换（整体流程 ↔ 细节机制） |
| **BertViz** | ~7,900 | 多视角可视化（head/model/neuron） |
| **AttentionViz** | ~160 | 全局模式理解（不只是局部 token） |

---

## 三、目标受众

### 3.1 主要受众

| 受众类型 | 特征 | 需求 |
|----------|------|------|
| **零基础学习者（核心）** | 非技术背景、对AI好奇但未系统学习 | 先建立直觉，再逐步进入机制 |
| **转岗/跨学科学习者** | 有学习动力，但数学与代码基础不稳定 | 需要低门槛路径和可操作练习 |
| **初级开发者** | 能看懂基础 Python 代码 | 希望把“理解”落地到最小实现 |

### 3.2 前置知识要求

- **最低**：零基础可学（不要求线性代数/微积分先修）
- **推荐**：会基础 Python 语法，可更顺利完成“从零实现”实践任务

---

## 四、内容体系架构

### 4.1 整体结构

``` 
AI原理科普体系
│
├── 序章：为什么我们需要理解生成式AI
│
├── 第一篇：生成式AI最小直觉
│   ├── 1.1 Token与Embedding
│   ├── 1.2 下一词预测到底在做什么
│   ├── 1.3 训练与推理的区别
│   └── 可视化：Token概率分布与采样变化
│
├── 第二篇：Transformer革命 ⭐【优先开发】
│   ├── 2.1 序列建模的上下文瓶颈
│   ├── 2.2 注意力机制的直觉
│   ├── 2.3 Self-Attention 详解
│   ├── 2.4 多头注意力
│   ├── 2.5 位置编码
│   ├── 2.6 Transformer完整架构
│   └── 可视化：Attention矩阵动态计算
│
├── 第三篇：从预训练到对齐
│   ├── 3.1 预训练：学习语言统计规律
│   ├── 3.2 指令微调（SFT）
│   ├── 3.3 RLHF与DPO
│   └── 可视化：训练阶段能力对比
│
├── 第四篇：规模的力量 - Scaling Law
│   ├── 4.1 什么是Scaling Law
│   ├── 4.2 参数量、数据量、计算量的关系
│   ├── 4.3 涌现能力：量变到质变
│   ├── 4.4 Chinchilla论文的启示
│   └── 可视化：Loss曲线随规模变化的动画
│
├── 第五篇：推理、解码与系统工程
│   ├── 5.1 Prompt与In-Context Learning
│   ├── 5.2 解码策略（Greedy/Top-k/Top-p/Temperature）
│   ├── 5.3 上下文窗口、KV Cache 与延迟
│   ├── 5.4 RAG与工具调用
│   └── 可视化：解码策略与系统瓶颈对比
│
├── 第六篇：多模态与Agent
│   ├── 6.1 视觉/语音/文本统一表示趋势
│   ├── 6.2 Agent循环：规划-执行-反思
│   ├── 6.3 工作流编排与工具生态
│   └── 可视化：能力边界概念图
│
├── 第七篇：安全、边界与未来
│   ├── 7.1 幻觉、偏见与可控性
│   ├── 7.2 对齐与治理
│   ├── 7.3 AI与人类协作
│   └── 可视化：AI发展时间线
│
└── 附录
    ├── 数学基础补充
    ├── 术语表
    ├── 推荐资源
    └── 实践练习
```

### 4.2 章节详细设计：Transformer（优先开发）

以**第二篇：Transformer革命**为例：

| 小节 | 核心概念 | 可视化内容 | 交互元素 |
|------|----------|------------|----------|
| 2.1 序列建模瓶颈 | 固定窗口与长上下文建模难题 | 上下文覆盖率可视化 | 调整上下文长度观察效果 |
| 2.2 注意力直觉 | "看哪里"的概念、权重分配 | 人眼注意力类比动画 | 点击不同区域看权重变化 |
| 2.3 Self-Attention | Q/K/V的概念、注意力计算 | 矩阵运算逐步动画 | 输入自己的句子看Attention |
| 2.4 多头注意力 | 多个子空间、并行计算 | 多头分叉与合并动画 | 切换不同头查看关注点 |
| 2.5 位置编码 | 正弦余弦编码、相对位置 | 位置编码矩阵热力图 | 拖动词位置观察编码变化 |
| 2.6 完整架构 | Decoder-only 主干、Layer Norm | 整体架构动态数据流 | 逐步展开各层 |

---

## 五、技术实现方案

### 5.1 技术栈（已确定）

```
核心框架：Astro 5.x + Starlight
交互组件：React 18
可视化库：D3.js
动画库：Framer Motion / Motion for React
样式方案：Tailwind CSS
内容格式：MDX + JSON
部署平台：GitHub Pages / Vercel
```

**选型理由**：
| 技术 | 理由 |
|------|------|
| **Astro + Starlight** | 专为内容网站设计，章节化内容快速上线，性能优异（零JS默认），SEO友好 |
| **React Islands** | 关键交互模块高质量实现，组件复用性强 |
| **D3.js** | 数据可视化标准库，灵活度高，与 Transformer Explainer 技术栈一致 |
| **Framer Motion** | 声明式动画API，科技感动效实现 |
| **Tailwind CSS** | 快速开发，暗色主题支持好，科技感配色 |

### 5.2 产品结构设计

采用**双模式**结构：

| 模式 | 描述 | 用户场景 |
|------|------|----------|
| **导学模式** | 章节化、线性学习 | 系统学习AI原理 |
| **实验模式** | 自由改参数和输入，立即看结果 | 探索和理解 |

**核心原则**：每章必须有"一个可操作实验"，否则理解停留在阅读层。

### 5.3 Transformer First 的落地方式（MVP核心）

**本轮MVP目标**：只做一个闭环，`从零理解 Transformer -> 从零实现最小可运行 Transformer`。

**第一优先交互模块**（按顺序实现）：

| 优先级 | 模块 | 功能 | 交互方式 |
|--------|------|------|----------|
| 1 | **Attention Explorer** | Token级热力图 | 输入文本，实时显示注意力权重 |
| 2 | **QKV Step-by-Step** | Q/K/V与缩放点积计算拆解 | 分步播放矩阵计算 |
| 3 | **Multi-Head Comparator** | 头之间差异可视化 | 切换不同 Head 查看关注点 |
| 4 | **Positional Encoding Toggle** | 有/无位置编码对比 | 开关切换，观察效果差异 |
| 5 | **Mini Transformer Lab** | 从零拼装最小模型并运行 | 按步骤完成代码与训练 |

**内容分层**：

| 层级 | 目标 | 内容深度 |
|------|------|----------|
| **基础层** | 零基础看懂现象 | 直观动画 + 类比解释 + 术语最小集 |
| **实现层** | 从零写出最小模型 | PyTorch 逐步搭建 + 训练/推理脚本 |

### 5.4 文件结构

```
ai-principles-viz/
├── docs/                    # PRD与设计文档
│   ├── PRD.md
│   └── AI-科普产品调研-GitHub-X-Codex.md
├── src/
│   ├── pages/               # Astro页面
│   ├── components/          # React组件
│   │   ├── viz/             # 可视化组件
│   │   │   ├── TransformerViz.tsx
│   │   │   ├── AttentionExplorer.tsx
│   │   │   ├── QKVStepByStep.tsx
│   │   │   └── MiniTransformerLab.tsx
│   │   └── ui/              # 通用UI组件
│   ├── content/             # MDX内容文件
│   │   ├── chapter-1/       # 生成式AI最小直觉
│   │   ├── chapter-2/       # Transformer（优先）
│   │   └── ...
│   └── lib/                 # 工具函数
├── public/
│   ├── animations/          # SVG动画文件
│   └── images/              # 静态图片
└── package.json
```

---

## 六、设计规范

### 6.1 视觉设计

**配色方案**（科技感）：

```css
:root {
  /* 主色调 */
  --primary-500: #6366f1;      /* 靛蓝 */
  --primary-400: #818cf8;

  /* 辅助色 */
  --secondary-400: #22d3ee;    /* 青色 */
  --secondary-500: #06b6d4;

  /* 强调色 */
  --accent-400: #f472b6;       /* 粉色 */
  --accent-500: #ec4899;

  /* 背景色 */
  --dark-900: #0f172a;         /* 深蓝黑 */
  --dark-950: #020617;

  /* 文字色 */
  --text-primary: #e2e8f0;
  --text-secondary: #94a3b8;

  /* 状态色 */
  --success: #4ade80;
  --warning: #fbbf24;
}
```

**字体**：
- 正文：Inter / system-ui
- 代码：JetBrains Mono / Fira Code

### 6.2 动画规范

| 参数 | 规范 |
|------|------|
| 持续时间 | 300ms - 1000ms（根据复杂度） |
| 缓动函数 | ease-out（入场）、ease-in（退场） |
| 循环动画 | 至少2秒一个周期 |
| 帧率 | 60fps |

### 6.3 交互设计规范

| 交互类型 | 使用场景 |
|----------|----------|
| **Hover提示** | 展示详细数据、公式 |
| **点击展开** | 深入解释、代码示例 |
| **滑块调节** | 参数变化（学习率、层数等） |
| **拖拽排序** | 序列、注意力权重调整 |
| **输入框** | 自定义文本输入 |

---

## 七、项目里程碑

### 7.1 阶段规划

| 阶段 | 内容 | 产出 | 时间 |
|------|------|------|------|
| **Phase 1** | Transformer认知底座（零基础） | 2.1-2.6 章节基础层 + 3个核心交互 | Week 1-2 |
| **Phase 2** | 从零实现最小Transformer ⭐ | Mini Transformer Lab + 可运行训练/推理示例 | Week 3-4 |
| **Phase 3** | MVP验收与发布 | 验收演示、文档化、公开预览 | Week 5 |
| **Phase 4** | 其他生成式专题扩展 | 预训练/对齐、Scaling、推理系统、多模态Agent | Week 6+ |

### 7.2 MVP范围

**最小可行产品（MVP）** 包含：

- ✅ 完整的网站框架（Astro + Starlight）
- ✅ 第二篇（Transformer核心部分）完整闭环（2.1-2.6）
- ✅ 5个核心交互模块（含 `Mini Transformer Lab`）
- ✅ 响应式设计
- ✅ 暗色主题（科技感）
- ✅ 首页 "Transformer First" 入口作为主 CTA
- ✅ 零基础导学路径（术语卡 + 10分钟预备知识）

### 7.3 MVP验收标准（已确认：C）

**验收通过定义**：学习者在零基础导学后，能够从零实现一个最小可运行 Transformer。

1. 代码实现完整：包含 Token Embedding、位置编码、自注意力、多头、FFN、残差+LayerNorm、输出头。
2. 训练结果可验证：在 toy 数据集训练后，loss 相比初始值下降至少 30%。
3. 推理结果可运行：给定 prompt 可自回归生成不少于 20 个 token。
4. 机制解释可对齐：学习者能将每个模块的“作用”与对应代码位置一一对应说明。

---

## 八、传播与增长策略

### 8.1 发布渠道

| 渠道 | 内容 |
|------|------|
| **GitHub** | 开源代码、Markdown源文件 |
| **个人网站** | 完整交互体验 |
| **X/Twitter** | 预告、演示片段、迭代进展 |
| **微信公众号** | 精选内容图文版 |
| **B站/YouTube** | 录屏讲解视频 |

### 8.2 X/Twitter 传播节奏

基于对 @3blue1brown、@karpathy 等账号的观察：

| 时间节点 | 动作 | 内容 |
|----------|------|------|
| 上线前 7 天 | 发布 3 条 teaser | 15-30 秒演示片段 |
| 上线日 | 发布 1 条主帖 | 一句价值主张 + demo link + gif |
| 上线后 2 周 | 每周 2 条短拆解 | "你以为 vs 实际机制" |
| 持续 | build-in-public | 每次迭代可视化展示变化 |

**高表现内容特征**：
1. 强钩子开场：一句话说明"你现在就能看到什么"
2. 直给链接：首帖就放 Demo/视频链接
3. 可视化优先：短视频/GIF 比纯文字效果好
4. 持续迭代：不只发一次上线公告

---

## 九、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| **交互炫但学习路径混乱** | 用户迷失 | 每个交互后强制有"我看到了什么 → 这意味着什么" |
| **章节太多导致每章都浅** | 质量不高 | 先保证 Transformer 章节成为标杆样章 |
| **发布节奏断档** | 用户流失 | 固定周更节奏，优先小步快跑可见成果 |
| **可视化开发成本高** | 进度延期 | 使用现成库、AI辅助 |
| **技术更新快** | 内容过时 | 设计可更新的架构 |
| **个人时间有限** | 难以持续 | 模块化，可独立发布 |

---

## 十、成功指标

| 指标 | 目标 | 衡量方式 |
|------|------|----------|
| **Transformer闭环完成度** | 2.1-2.6 全部交付 | 章节完成度 |
| **最小实现通过率** | >= 70% 学习者完成最小实现 | Lab 提交/演示记录 |
| **核心交互模块完成数** | >= 5 | 组件清单 |
| **理解测验达标率** | >= 75% | 章节测验 |
| **MVP用户反馈** | NPS > 40 | 访谈与问卷 |

---

## 十一、7天行动计划

基于调研结论，立即可执行的 7 天动作：

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1 | 锁定零基础导学结构与术语卡 | 2.1-2.6 基础层大纲 + 术语最小集 |
| Day 2 | 实现 Attention Explorer + QKV 分步演示 | 2 个可交互模块可运行 |
| Day 3 | 实现 Multi-Head/Positional Encoding 组件 | 2 个对比模块可运行 |
| Day 4 | 搭建 `Mini Transformer Lab` 代码骨架 | 可运行的模型 skeleton |
| Day 5 | 完成 toy 数据训练与生成脚本 | loss 下降曲线 + 文本生成结果 |
| Day 6 | 串联“概念 -> 代码 -> 运行结果”学习路径 | MVP 学习闭环页 |
| Day 7 | 按 7.3 标准执行验收并发布预览版 | 验收记录 + 预览链接 |

---

## 附录A：术语表

| 术语 | 定义 | 可视化关联 |
|------|------|------------|
| 注意力 | 分配计算资源的机制 | 权重热力图 |
| Token | 文本的最小处理单位 | 分词可视化 |
| Self-Attention | 序列内元素相互关注的机制 | Q/K/V 矩阵动画 |
| Multi-Head | 并行多个注意力子空间 | 多头分叉可视化 |
| 位置编码 | 为序列注入位置信息 | 正弦波热力图 |
| 预训练 | 基于海量语料进行下一词预测学习 | Loss 曲线图 |
| 对齐 | 让模型输出更符合人类偏好与安全约束 | 对齐前后对比图 |
| Scaling Law | 模型性能与规模的幂律关系 | Loss 曲线图 |

---

## 附录B：参考资源

### 标杆项目

| 项目 | 链接 | 说明 |
|------|------|------|
| Transformer Explainer | https://github.com/poloclub/transformer-explainer | 核心对标项目 |
| TensorFlow Playground | https://playground.tensorflow.org/ | 即时反馈设计 |
| Diffusion Explainer | https://github.com/poloclub/diffusion-explainer | 多层抽象设计 |
| BertViz | https://github.com/jessevig/bertviz | Attention可视化 |

### 官方论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - Kaplan et al.
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - InstructGPT

### 优质科普

- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Andrej Karpathy - Zero to Hero](https://karpathy.ai/zero.hero/)
- [Jay Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [李宏毅机器学习课程](https://speech.ee.ntu.edu.tw/~hylee/ml/)
- [Distill.pub - Communicating with Interactive Articles](https://distill.pub/2020/communicating-with-interactive-articles)

### 技术工具

- [Manim](https://www.manim.community/) - 数学动画引擎
- [D3.js](https://d3js.org/) - 数据可视化库
- [Observable](https://observablehq.com/) - 交互式笔记本

---

## 附录C：GLM

### 关于GLM模型

GLM（General Language Model）是智谱AI开发的大语言模型系列。本项目的部分内容规划参考了GLM系列模型的技术报告，包括：

- **GLM-130B**：开源双语（中英）双向稠密模型
- **ChatGLM**：针对中文对话优化的模型系列
- **GLM-4**：最新一代多模态大模型

### GLM的技术特点

| 特性 | 说明 |
|------|------|
| **架构** | 基于Transformer的自回归空白填充模型 |
| **训练** | 多任务预训练 + 指令微调 + RLHF |
| **能力** | 文本生成、代码、多模态理解 |
| **开源** | 部分模型权重开源，可本地部署 |

### 在本项目中的应用

本项目在讲解以下主题时，可以引用GLM作为案例：

1. **多语言能力**：GLM的双语训练策略
2. **指令微调**：ChatGLM的对话优化
3. **高效推理**：量化技术在GLM中的应用
4. **开源生态**：GLM对开发者社区的支持

---

> **文档状态**：v2.1 完成（已纳入零基础 + 从零实现验收标准）
> **下一步**：执行 7 天行动计划，优先实现 Transformer MVP
> **联系方式**：[待补充]

---

*本文档由 Claude (Anthropic) 协助创建。*
