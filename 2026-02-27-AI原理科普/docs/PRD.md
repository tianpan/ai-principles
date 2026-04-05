# 生成式AI原理科普项目 PRD（统一最终版）

> **文档ID**：PRD-GenAI-Final  
> **版本**：v3.0-final  
> **创建日期**：2026-02-28  
> **最后更新**：2026-02-28  
> **文档状态**：Single Source of Truth（唯一主PRD）  
> **融合来源**：`docs/PRD.md`、`docs/AI-科普全景-PRD-Codex.md`、`docs/AI-科普产品调研-GitHub-X-Codex.md`

---

## 1. 融合决策与取舍

### 1.1 冲突裁决标准（用于后续迭代）

1. 与已确认范围一致优先（生成式AI优先于全量扩展）。
2. 可验证学习结果优先（可运行实现优先于纯概念描述）。
3. 交付确定性优先（可在 5 周内闭环优先于大而全方案）。
4. 零基础可达性优先（先可理解，再逐步进阶）。

| 决策点 | 候选方案 | 最终选择 | 选择理由 |
|------|------|------|------|
| 项目范围 | 全AI（含CNN/RNN） vs 生成式AI主线 | **生成式AI主线** | 与你最新明确范围一致，避免内容发散 |
| 上线策略 | 全章节骨架先行 vs Transformer闭环先行 | **Transformer MVP先行** | 最快形成可验证学习闭环 |
| 目标用户 | 泛技术人群 vs 零基础优先 | **零基础优先 + 实现层进阶** | 兼顾可懂与可做 |
| 验收标准 | 概念理解为主 vs 可运行实现 | **可从零实现最小Transformer（C）** | 可客观验收，学习结果可证明 |
| 指标体系 | 流量/Star优先 vs 学习效果优先 | **学习效果优先，传播指标辅助** | 与项目本质目标一致 |
| 视觉规范 | 泛科技感描述 vs 具体设计约束 | **保留可执行视觉规范** | 降低设计偏差，提高一致性 |

---

## 2. 项目定位

### 2.1 项目目标

打造一个面向零基础学习者的生成式AI交互科普产品，让用户从“看懂Transformer机制”走到“从零实现一个最小可运行Transformer”。

### 2.2 范围（In Scope）

1. Transformer 核心机制（Q/K/V、Self-Attention、多头、位置编码、残差+LayerNorm、FFN）。
2. 预训练、指令微调、对齐（RLHF/DPO）主流程。
3. Scaling Law 核心直觉与工程含义。
4. 推理与解码（Greedy/Top-k/Top-p/Temperature）、上下文窗口、KV Cache、RAG入门。
5. 多模态与Agent的框架化认知（不做深工程实现）。

### 2.3 非目标（Out of Scope）

1. 不展开 CNN/RNN/LSTM/GRU 等早期架构专题。
2. 不做完整数学证明体系（仅保留最小必要公式）。
3. 不做完整课程平台能力（考试、社交、学习打卡）。
4. 不以模型榜单追新为主线。

---

## 3. 目标用户与学习结果

### 3.1 目标用户

1. 零基础学习者（核心）。
2. 转岗/跨学科学习者。
3. 能看懂基础 Python 的初级开发者。

### 3.2 学习结果定义

1. 概念层：能解释 Transformer 各模块“在做什么、为什么需要”。
2. 计算层：能手动走通一次简化 attention 计算流程。
3. 实现层（核心）：能从零实现最小可运行 Transformer 并完成训练与生成演示。

---

## 4. 产品形态与信息架构

### 4.1 双模式结构

1. 导学模式：章节化线性学习。
2. 实验模式：参数可调、实时反馈。

### 4.2 章节主线（生成式AI）

1. Chapter 0：导航与学习路径。
2. Chapter 1：生成式AI最小直觉（Token/Embedding/训练vs推理）。
3. Chapter 2：Transformer 核心机制（MVP核心）。
4. Chapter 3：从预训练到对齐（SFT、RLHF、DPO）。
5. Chapter 4：Scaling Law 与涌现。
6. Chapter 5：推理、解码与系统工程（KV Cache、RAG）。
7. Chapter 6：多模态与Agent。
8. Chapter 7：安全、边界与未来。

### 4.3 页面结构

1. 首页：Transformer First 主入口 + 学习路径。
2. 章节页：概念解释 + 动态演示 + 小测。
3. 实验页：Attention/解码/Scaling 等统一实验台。
4. 术语页：可检索术语与概念关系图。

---

## 5. MVP定义（Transformer First）

### 5.1 MVP必须交付

1. Chapter 2（2.1-2.6）完整闭环内容。
2. 五个核心交互模块：
   - Attention Explorer
   - QKV Step-by-Step
   - Multi-Head Comparator
   - Positional Encoding Toggle
   - Mini Transformer Lab
3. 零基础导学路径（术语卡 + 10分钟预备知识）。
4. 可公开访问的 Web 版本（桌面端与移动端可用）。

### 5.2 MVP验收标准（采用 C）

1. 代码实现完整：包含 Embedding、位置编码、自注意力、多头、FFN、残差+LayerNorm、输出头。
2. 训练结果可验证：toy 数据集训练后，loss 较初始下降不少于 30%。
3. 推理结果可运行：给定 prompt 可自回归生成不少于 20 个 token。
4. 机制解释可对齐：学习者能将模块作用与代码位置一一对应说明。

---

## 6. 设计与交互规范

### 6.1 设计原则

1. 先直觉后细节。
2. 一次只解释一个变量。
3. 每个互动模块必须有“我看到了什么 -> 这意味着什么”总结框。

### 6.2 视觉方向

1. 深空科技风：深蓝灰基底 + 青色主强调 + 绿色辅助。
2. 字体建议：`Space Grotesk`（标题）、`IBM Plex Sans`（正文）、`JetBrains Mono`（代码）。
3. 使用 Design Tokens 统一颜色、间距、圆角、阴影、动效曲线。

### 6.3 动效约束

1. 页面入场 300-500ms。
2. 单页面关键动效不超过 3 类。
3. 动效服务理解，不做过载炫技。

---

## 7. 技术方案

### 7.1 MVP技术栈

1. Astro + Starlight（内容骨架）。
2. React + TypeScript（交互组件，Islands 按需加载）。
3. D3.js（可视化主力）。
4. Motion for React / GSAP（动效）。
5. Tailwind CSS + CSS Variables（样式与 Token）。
6. MDX + JSON（内容与交互配置分离）。
7. Vercel / GitHub Pages（部署）。

### 7.2 实施原则

1. 先打通 Transformer 闭环，再扩展章节。
2. 动态组件模块化复用。
3. 首版本地静态数据，后端后置。

---

## 8. 里程碑

| 阶段 | 周期 | 目标产出 |
|------|------|------|
| Phase 1 | Week 1-2 | Chapter 2 基础层 + 3 个核心交互 |
| Phase 2 | Week 3-4 | Mini Transformer Lab + 可运行训练/推理示例 |
| Phase 3 | Week 5 | 按 C 标准完成验收并发布公开预览 |
| Phase 4 | Week 6+ | 扩展到预训练/对齐、Scaling、系统工程、多模态Agent |

---

## 9. 成功指标

1. Transformer闭环完成度：Chapter 2（2.1-2.6）100%交付。
2. 最小实现通过率：>= 70% 学习者完成最小实现。
3. 理解测验达标率：>= 75%。
4. 交互有效性：动态模块停留时长 >= 静态页面 1.5 倍。
5. MVP用户反馈：NPS > 40。

---

## 10. 风险与应对

1. 风险：交互炫但学习路径混乱。  
   应对：每个交互统一“现象-解释-代码映射”结构。
2. 风险：范围回弹导致失焦。  
   应对：严格遵守本PRD范围，不新增 CNN/RNN 专题。
3. 风险：实现难度高导致延期。  
   应对：先保底最小模型，再迭代高级功能。
4. 风险：发布节奏中断。  
   应对：固定周更，优先发布可见小成果。

---

## 11. 7天执行计划（对齐MVP）

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1 | 锁定 Chapter 2 信息架构与术语卡 | 2.1-2.6 结构稿 |
| Day 2 | 完成 Attention Explorer + QKV Step-by-Step | 2 个可运行模块 |
| Day 3 | 完成 Multi-Head + Positional Encoding | 2 个可运行模块 |
| Day 4 | 搭建 Mini Transformer Lab 骨架 | 可运行 skeleton |
| Day 5 | 完成 toy 训练与生成脚本 | loss 曲线 + 生成样例 |
| Day 6 | 串联“概念 -> 代码 -> 运行结果”闭环页 | MVP学习闭环 |
| Day 7 | 按 C 标准验收并发布预览 | 验收记录 + 预览链接 |

---

## 12. 参考来源（融合保留）

1. https://github.com/poloclub/transformer-explainer  
2. https://poloclub.github.io/transformer-explainer/  
3. https://github.com/poloclub/diffusion-explainer  
4. https://github.com/tensorflow/playground  
5. https://github.com/jessevig/bertviz  
6. https://arxiv.org/abs/1706.03762  
7. https://arxiv.org/abs/2203.02155  
8. https://distill.pub/2020/communicating-with-interactive-articles

---

> 本文件为当前唯一主PRD。其他 PRD/调研文件保留为历史输入，不再作为需求基线。
