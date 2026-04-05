## ADDED Requirements

### Requirement: Positional Encoding Toggle 组件
系统 SHALL 提供 Positional Encoding Toggle 组件，对比有无位置编码的效果。

#### Scenario: 切换位置编码
- **WHEN** 用户开关位置编码
- **THEN** 系统展示有无位置编码的 token 表示差异

### Requirement: sin/cos 可视化
系统 SHALL 可视化展示 sin/cos 位置编码的波形。

#### Scenario: 波形展示
- **WHEN** 用户查看位置编码
- **THEN** 显示不同位置的 sin/cos 波形图
