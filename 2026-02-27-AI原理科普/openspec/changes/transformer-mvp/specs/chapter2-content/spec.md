## ADDED Requirements

### Requirement: Chapter 2 完整内容
系统 SHALL 提供 Chapter 2（Transformer 核心机制）的完整内容，包含 2.1-2.6 六个小节。

#### Scenario: 章节导航完整
- **WHEN** 用户访问 Chapter 2
- **THEN** 可看到 2.1 Token/Embedding 到 2.6 FFN+输出层的完整导航

### Requirement: 概念三层递进
每个知识点 SHALL 按"概念 → 计算 → 实现"三层递进组织。

#### Scenario: 知识点结构
- **WHEN** 用户阅读某个知识点
- **THEN** 可依次看到直觉解释、计算过程、代码示例

### Requirement: 术语卡支持
系统 SHALL 为每个专业术语提供可点击的术语卡。

#### Scenario: 术语卡展示
- **WHEN** 用户点击术语
- **THEN** 弹出术语定义和简单解释
