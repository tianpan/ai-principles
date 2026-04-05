## ADDED Requirements

### Requirement: Project resolution by role
系统 SHALL 根据当前登录用户角色自动解析项目：班组长自动匹配其负责的项目，其他角色显示项目选择器。

#### Scenario: Team leader login
- **WHEN** 班组长用户登录并进入功能页面
- **THEN** 系统自动加载该班组长负责的项目，页面顶部显示项目名称

#### Scenario: Manager login
- **WHEN** 项目经理或管理员登录并进入功能页面
- **THEN** 页面顶部显示项目选择下拉框，默认选中第一个项目

### Requirement: Project selector component
系统 SHALL 提供 `ProjectSelector` 组件，接受 `projects` 列表和 `onSelect` 回调。

#### Scenario: Selector renders
- **WHEN** 组件接收到多个项目
- **THEN** 渲染为下拉选择框，显示所有项目名称，当前选中项高亮

#### Scenario: User switches project
- **WHEN** 用户在下拉框中选择另一个项目
- **THEN** `onSelect` 回调触发，返回选中的项目对象，页面数据刷新为新项目数据
