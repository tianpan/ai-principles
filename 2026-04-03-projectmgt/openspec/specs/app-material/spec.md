## MODIFIED Requirements

### Requirement: Material page role adaptation
材料设备页面 SHALL 支持所有角色使用。

#### Scenario: Manager uses material page
- **WHEN** 项目经理进入材料设备页面
- **THEN** 显示项目选择器，选择项目后可操作领料/退料/设备/采买

### Requirement: Material quick fill
材料设备页面的领料/退料 tab SHALL 提供常用材料快捷选择和自动单号生成。

#### Scenario: Quick fill material name
- **WHEN** 用户点击材料名称旁的"+"按钮
- **THEN** 弹出常用材料选择面板，选择后自动填充名称和规格

#### Scenario: Auto requisition number
- **WHEN** 用户添加新的材料行
- **THEN** 自动生成领料单号，格式 MR-YYYYMMDD-NNN

### Requirement: Form validation
材料设备页面 SHALL 验证每行材料的名称和数量不为空。

#### Scenario: Submit with empty row
- **WHEN** 用户提交时存在材料名称为空或数量为 0 的行
- **THEN** 显示 Toast 提示"请填写完整的材料信息"，不提交
