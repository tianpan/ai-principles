## ADDED Requirements

### Requirement: Common material library
系统 SHALL 在材料名称输入框旁提供"+"按钮，点击后弹出常用材料选择面板。

#### Scenario: User opens material selector
- **WHEN** 用户点击材料名称旁的"+"按钮
- **THEN** 弹出常用材料列表：DN100镀锌钢管、DN200无缝钢管、PE管DN90、PE管DN200、焊条E4303、球阀DN100、法兰DN200、弯头90°

#### Scenario: User selects a material
- **WHEN** 用户从面板中选择"DN100镀锌钢管"
- **THEN** 材料名称自动填入"DN100镀锌钢管"，规格自动填入"DN100"，光标移到数量输入框

### Requirement: Auto-generated requisition number
领料和退料 tab 的每行 SHALL 自动生成领料单号，格式 `MR-YYYYMMDD-NNN`。

#### Scenario: New row added
- **WHEN** 用户在领料 tab 点击"添加行"
- **THEN** 新行自动填充领料单号，如 `MR-20260404-001`，序号递增

### Requirement: Equipment preset options
设备 tab 的设备类型 SHALL 使用下拉选择，预设选项：吊车、挖掘机、压路机、电焊机、发电机。

#### Scenario: Equipment type selection
- **WHEN** 用户在设备 tab 点击设备类型字段
- **THEN** 弹出下拉选项：吊车、挖掘机、压路机、电焊机、发电机

#### Scenario: Equipment type selected
- **WHEN** 用户选择"吊车"
- **THEN** 设备类型填入"吊车"，规格字段可输入（如"25吨"），台班字段高亮
