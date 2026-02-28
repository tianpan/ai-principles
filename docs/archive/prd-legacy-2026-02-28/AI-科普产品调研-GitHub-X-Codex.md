# AI 科普产品调研报告（GitHub + X/Twitter）- Codex

## 1. 调研目标与范围
本次调研聚焦两类问题：
1. GitHub 上高质量 AI 科普产品是如何设计交互、内容结构与技术架构的。
2. X/Twitter 上这类产品如何发布、传播与迭代。

调研方法：
1. 以 Transformer/Attention/Diffusion/Neural Network Playground 相关关键词检索。
2. 优先采样开源仓库主页、官方演示站点、论文与项目说明。
3. 对 X/Twitter 侧，采样公开可访问的账号/帖子镜像页面（TwStalker）观察传播方式。

---

## 2. GitHub 代表项目与可借鉴点

### A. 强交互 explainer（最贴近你的目标）
1. `poloclub/transformer-explainer`（约 6.8k stars）
   - 核心做法：在浏览器内运行实时 GPT-2，用户输入文本后可直接观察 Transformer 内部计算和预测变化。
   - 可借鉴：
     1. “概念解释 + 实时模型 + 参数调节”三合一。
     2. 明确提供 Live Demo、论文、视频三入口。

2. `poloclub/diffusion-explainer`（约 451 stars）
   - 核心做法：用交互动画讲清 Stable Diffusion 从提示词到图像的过程，强调“无需安装、无需 GPU”。
   - 可借鉴：
     1. 非技术用户友好（零门槛访问）。
     2. 多层抽象切换（整体流程 <-> 细节机制）。

3. `poloclub/cnn-explainer`（约 8.9k stars）
   - 核心做法：明确面向 non-experts，交互化拆解 CNN。
   - 可借鉴：
     1. 用户定位清晰，术语和节奏严格面向初学者。
     2. 学术严谨 + 产品体验并重。

### B. 经典“可玩”学习工具
4. `tensorflow/playground`（约 12.8k stars）
   - 核心做法：最小操作闭环（改参数 -> 立刻看到决策边界和 loss 变化）。
   - 可借鉴：
     1. 控件极简、反馈即时。
     2. 颜色语义、图层语义长期一致，学习成本低。

5. `PAIR-code/what-if-tool`（约 990 stars）
   - 核心做法：强调“不同角色都可用”（研究者、业务方、普通用户）。
   - 可借鉴：
     1. 同一产品支持多角色解释深度。
     2. 不止“看图”，还支持模型比较、反事实编辑。

### C. Transformer 内部可视化工具
6. `jessevig/bertviz`（约 7.9k stars）
   - 核心做法：在 Jupyter/Colab 中快速可视化 attention。
   - 可借鉴：
     1. 研究/教学两用。
     2. 多视角（head/model/neuron）提升解释深度。

7. `catherinesyeh/attention-viz`（约 160 stars）
   - 核心做法：关注 query-key 全局关系，不只局部热力图。
   - 可借鉴：
     1. 从“局部 token 注意力”升级到“全局模式理解”。

8. `bhoov/exbert`
   - 核心做法：探索 Transformer 表征与注意力的可视分析工具。
   - 可借鉴：
     1. 支持“实验探索型学习”，而不只是线性阅读。

---

## 3. X/Twitter 侧传播方式观察

### 3.1 高表现内容的共同特征
1. 强钩子开场：一句话说明“你现在就能看到什么”。
2. 直给链接：首帖就放 Demo/视频链接，降低跳转成本。
3. 可视化优先：短视频/GIF 比纯文字效果更好。
4. build-in-public：持续发迭代进展，而非只发一次上线公告。

### 3.2 典型样本（可见公开页面）
1. `@3blue1brown`：新视频发布帖通常是“主题一句话 + 直链 + 视觉片段”，互动量高。
2. `@karpathy`：长帖拆解技术直觉和工作流，形成高密度讨论。
3. `@alec_helbling`：展示 ManimML 时，强调“工具用途 + 与生态关系 + 仓库链接”，传播效率高。
4. `@manim_community`：持续转发创作者案例，形成内容生态而不是单点爆发。

说明：X/Twitter 数据以公开网页镜像可见内容为样本，互动数字随时间变化。

---

## 4. 对你项目最关键的可执行结论

### 4.1 产品结构
1. 采用“双模式”结构：
   - 导学模式：章节化、线性学习。
   - 实验模式：自由改参数和输入，立即看结果。
2. 每章必须有“一个可操作实验”，否则理解停留在阅读层。

### 4.2 Transformer First 的落地方式
1. 第一优先交互模块：
   1. Attention Explorer（token 级热力图）。
   2. Multi-Head Comparator（头之间差异）。
   3. Positional Encoding Toggle（有/无位置编码对比）。
   4. Decoding Lab（temperature/top-k/top-p 实时对比）。
2. 内容层采用“基础层/进阶层”切换：
   1. 基础层：看懂现象。
   2. 进阶层：看懂计算与工程约束。

### 4.3 传播与增长（X/Twitter）
1. 上线前 7 天：发布 3 条 teaser（15-30 秒演示片段）。
2. 上线日：发布 1 条主帖（一句价值主张 + demo link + gif）。
3. 上线后 2 周：每周 2 条“你以为 vs 实际机制”短拆解。
4. 每次迭代都要可视化展示变化，不只写 changelog。

---

## 5. 技术选型校准（结合调研）
结合调研结论，当前你的技术路线建议保持为：
1. `Astro + Starlight`：保证章节化内容快速上线。
2. `React Islands + D3`：保证关键交互模块高质量实现。
3. `Motion for React/GSAP`：服务“科技感”动效，不做过载炫技。
4. `MDX + JSON`：内容与交互配置分离，便于持续迭代。

---

## 6. 风险与规避
1. 风险：交互做得炫但学习路径混乱。
   - 规避：每个交互后强制有“我看到了什么 -> 这意味着什么”。
2. 风险：章节太多导致每章都浅。
   - 规避：先保证 Chapter 3（Transformer）成为标杆样章。
3. 风险：发布节奏断档。
   - 规避：固定周更节奏，优先小步快跑可见成果。

---

## 7. 你现在就能执行的 7 天动作
1. 完成 Chapter 3 页面 IA（信息架构）和四个交互组件占位。
2. 先实现 Attention Explorer 的最小闭环。
3. 录制 15 秒 demo gif，准备首条 X/Twitter 预告。
4. 设置统一视觉 Token（科技感色板/字体/动效曲线）。
5. 在首页放置“Transformer First”入口，作为主 CTA。

---

## 参考来源（精选）
### GitHub / 项目站点
1. https://github.com/poloclub/transformer-explainer
2. https://poloclub.github.io/transformer-explainer/
3. https://github.com/poloclub/diffusion-explainer
4. https://github.com/poloclub/cnn-explainer
5. https://github.com/tensorflow/playground
6. https://playground.tensorflow.org/
7. https://github.com/jessevig/bertviz
8. https://github.com/PAIR-code/what-if-tool
9. https://github.com/catherinesyeh/attention-viz
10. https://github.com/bhoov/exbert
11. https://mechanicalai.github.io/ganlab/

### 论文 / 方法论
12. https://arxiv.org/abs/2408.04619
13. https://arxiv.org/abs/2305.03509
14. https://distill.pub/2020/communicating-with-interactive-articles

### X/Twitter 公开样本页面（镜像）
15. https://twstalker.com/karpathy
16. https://mobile.twstalker.com/3blue1brown
17. https://ww.twstalker.com/goudals_s/status/1620082230063415297
18. https://ww.twstalker.com/manim_community
