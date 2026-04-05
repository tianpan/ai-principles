## ADDED Requirements

### Requirement: Multi-Head Comparator 组件
系统 SHALL 提供 Multi-Head Comparator 组件，对比单头和多头的差异。

#### Scenario: 单头 vs 多头对比
- **WHEN** 用户切换头数
- **THEN** 系统并排展示单头和多头的计算结果

### Requirement: 头数可调
系统 SHALL 允许用户调整注意力头数（1-8）。

#### Scenario: 头数调整
- **WHEN** 用户拖动滑块调整头数
- **THEN** 可视化实时更新
