## MODIFIED Requirements

### Requirement: Safety attendance photo capture
安全签到页面 SHALL 使用 PhotoCapture 组件替代静态占位符，拍照完成后才能进入下一步。

#### Scenario: Photo required for next step
- **WHEN** 用户在 Step 0 未拍照
- **THEN** "下一步"按钮保持禁用状态

#### Scenario: Photo taken enables next step
- **WHEN** 用户完成模拟拍照且至少选择1位签到成员
- **THEN** "下一步"按钮启用

### Requirement: Safety attendance role adaptation
安全签到页面 SHALL 支持所有角色使用。非班组长用户显示项目选择器。

#### Scenario: Manager views safety page
- **WHEN** 项目经理或管理员进入安全签到页面
- **THEN** 页面顶部显示项目选择下拉框，选择项目后加载该项目成员作为签到列表

#### Scenario: Team leader views safety page
- **WHEN** 班组长进入安全签到页面
- **THEN** 自动加载该班组长负责的项目成员，无需手动选择
