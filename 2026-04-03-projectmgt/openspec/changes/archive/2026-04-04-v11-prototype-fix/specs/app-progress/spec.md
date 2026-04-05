## MODIFIED Requirements

### Requirement: Progress data refresh after hydration
施工进度页面 SHALL 在 hydration 完成后正确显示工程量数据，不出现空白列表。

#### Scenario: Progress page loads after login
- **WHEN** 用户登录后导航到施工进度页面
- **THEN** 页面正确显示当前项目的所有工序和累计进度数据

#### Scenario: Progress page refreshes data
- **WHEN** hydration 完成且 currentUser 可用
- **THEN** 工程量列表从 mock 数据加载，每项显示工序名称、目标量、累计完成量、进度条

### Requirement: Progress page role adaptation
施工进度页面 SHALL 支持所有角色使用。

#### Scenario: Manager views progress
- **WHEN** 项目经理进入施工进度页面
- **THEN** 显示项目选择器，选择项目后展示该项目的工序列表和进度
