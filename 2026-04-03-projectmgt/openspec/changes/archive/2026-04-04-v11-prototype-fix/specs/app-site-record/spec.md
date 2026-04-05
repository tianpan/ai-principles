## MODIFIED Requirements

### Requirement: Site record role adaptation
施工记录页面 SHALL 支持所有角色使用。

#### Scenario: Manager creates site record
- **WHEN** 项目经理进入施工记录页面
- **THEN** 显示项目选择器，选择项目后可填写施工记录

### Requirement: Site record form validation
施工记录页面 SHALL 验证至少选择了记录类型和填写了位置信息。

#### Scenario: Submit without record type
- **WHEN** 用户未选择记录类型就点击提交
- **THEN** 显示 Toast 提示"请选择记录类型"，不提交

#### Scenario: Submit without location
- **WHEN** 用户选择记录类型但未填写位置
- **THEN** 显示 Toast 提示"请填写施工位置"，不提交
