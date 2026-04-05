## ADDED Requirements

### Requirement: 最小 Transformer 实现
系统 SHALL 提供可运行的最小 Transformer 实现代码。

#### Scenario: 代码可运行
- **WHEN** 用户运行代码
- **THEN** 模型可正常初始化和前向传播

### Requirement: toy 数据集训练
系统 SHALL 提供 toy 数据集和训练脚本。

#### Scenario: 训练 loss 下降
- **WHEN** 用户运行训练脚本
- **THEN** loss 较初始下降 ≥30%

### Requirement: 自回归推理
系统 SHALL 提供自回归推理脚本。

#### Scenario: 生成 token
- **WHEN** 用户输入 prompt
- **THEN** 系统生成 ≥20 个 token

### Requirement: 代码与模块对应
系统 SHALL 提供"代码位置 → 模块作用"的映射说明。

#### Scenario: 代码映射
- **WHEN** 用户查看代码
- **THEN** 可看到每个代码块对应的 Transformer 模块
