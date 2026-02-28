## 1. 项目基础设施

- [x] 1.1 初始化 Astro + Starlight 项目 (`npm create astro@latest`)
- [x] 1.2 配置 Tailwind CSS + CSS Variables (Design Tokens)
- [x] 1.3 创建项目目录结构 (`src/components/`, `src/content/`, `src/styles/`)
- [x] 1.4 配置 React Islands 支持
- [x] 1.5 设置深空科技风主题配色（深蓝灰 + 青色主强调 + 绿色辅助）

## 2. 内容骨架

- [x] 2.1 创建 Chapter 2 导航结构（2.1-2.6 六个小节）
- [x] 2.2 编写 2.1 Token/Embedding 内容（概念 → 计算 → 实现）
- [x] 2.3 编写 2.2 Self-Attention 内容
- [x] 2.4 编写 2.3 Multi-Head Attention 内容
- [x] 2.5 编写 2.4 Positional Encoding 内容
- [x] 2.6 编写 2.5 残差 + LayerNorm 内容
- [x] 2.7 编写 2.6 FFN + 输出层内容
- [x] 2.8 实现术语卡组件 (TerminologyCard)

## 3. 交互组件（按复杂度递增）

- [x] 3.1 实现 Positional Encoding Toggle 组件（sin/cos 波形可视化）
- [x] 3.2 实现 QKV Step-by-Step 组件（分步展示计算）
- [x] 3.3 实现 Attention Explorer 组件（Q/K/V 矩阵交互）
- [x] 3.4 实现 Multi-Head Comparator 组件（单头 vs 多头对比）
- [x] 3.5 实现通用 SummaryBox 组件（"我看到了什么 → 这意味着什么"）

## 4. Mini Transformer Lab

- [x] 4.1 编写最小 Transformer 模型代码（Embedding → FFN → 输出）
- [x] 4.2 实现 Self-Attention 层（Q/K/V 计算）
- [x] 4.3 实现多头注意力层
- [x] 4.4 实现位置编码（sin/cos）
- [x] 4.5 实现残差连接 + LayerNorm
- [x] 4.6 创建代码与模块映射说明文档

## 5. 训练与推理

- [x] 5.1 准备 toy 数据集（莎士比亚片段或简单对话）
- [x] 5.2 编写训练脚本（AdamW + CrossEntropyLoss）
- [x] 5.3 验证训练 loss 下降 ≥30%
- [x] 5.4 编写自回归推理脚本
- [x] 5.5 验证生成 token ≥20 个
- [x] 5.6 创建 Colab 运行版本

## 5.5 测试套件

- [x] 5.5.1 创建单元测试 (test_units.py) - 18 个测试用例
- [x] 5.5.2 创建集成测试 (test_integration.py) - 13 个测试用例
- [x] 5.5.3 创建 E2E 测试 (playwright) - Web 应用测试

## 6. 整合与验收

- [x] 6.1 串联"概念 → 代码 → 运行结果"闭环页
- [x] 6.2 实现零基础导学路径（首页入口 + 预备知识页）
- [x] 6.3 添加学习进度指示（侧边栏）
- [x] 6.4 响应式适配（桌面端 + 移动端）
- [x] 6.5 按 C 标准验收（4 项验收标准）
- [x] 6.6 部署到 GitHub Pages
- [x] 6.7 发布预览链接（已推送到 GitHub，自动部署中）
