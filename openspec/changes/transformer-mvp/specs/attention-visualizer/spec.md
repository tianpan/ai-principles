## ADDED Requirements

### Requirement: Attention Explorer 组件
系统 SHALL 提供 Attention Explorer 交互组件，可视化展示 Self-Attention 计算过程。

#### Scenario: 可视化 Q/K/V 计算
- **WHEN** 用户输入 Q、K、V 矩阵
- **THEN** 系统动态展示 attention score 计算过程

### Requirement: QKV Step-by-Step 组件
系统 SHALL 提供 QKV Step-by-Step 组件，逐步展示 Query、Key、Value 的计算。

#### Scenario: 分步展示
- **WHEN** 用户点击"下一步"
- **THEN** 系统展示当前步骤的计算过程和结果

### Requirement: 交互总结框
每个交互组件 SHALL 包含"我看到了什么 → 这意味着什么"总结框。

#### Scenario: 总结框展示
- **WHEN** 用户完成交互
- **THEN** 显示交互结果的总结解释
