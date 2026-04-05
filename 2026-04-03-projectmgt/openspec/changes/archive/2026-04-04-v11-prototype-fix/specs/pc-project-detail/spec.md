## ADDED Requirements

### Requirement: PC project detail page
系统 SHALL 在 `/pc/projects/[id]` 路由提供项目详情页，展示项目完整信息。

#### Scenario: Navigate to project detail
- **WHEN** 用户在 PC 项目列表点击某个项目行
- **THEN** 路由跳转到 `/pc/projects/:id`，渲染项目详情页

### Requirement: Project basic info section
项目详情页 SHALL 展示基本信息卡片：项目名称、所属部门、班组长、工期、状态、备注。

#### Scenario: Basic info renders
- **WHEN** 项目详情页加载
- **THEN** 显示项目名称、部门名称（从 departments 查询）、班组长姓名（从 users 查询）、起止日期、状态 badge、备注文字

### Requirement: Work items progress section
项目详情页 SHALL 展示工程量清单表格，每行显示：工序名称、目标量、单位、权重、累计完成量、进度条百分比。

#### Scenario: Progress data displays
- **WHEN** 项目有 workItems 和 daily records
- **THEN** 每个工序的累计完成量通过 `getCumulativeProgress(projectId, workItemId)` 计算，进度条 = 累计/目标

### Requirement: Recent daily records section
项目详情页 SHALL 展示近 7 天的日报列表，按日期倒序，每行显示：日期、模块类型、状态、操作链接。

#### Scenario: Records list renders
- **WHEN** 项目有 daily records
- **THEN** 列表显示该项目的所有记录，按日期倒序排列，状态用不同颜色 badge 标识

### Requirement: Project members section
项目详情页 SHALL 展示项目成员表格：姓名、岗位、资质证书、有效期。

#### Scenario: Members list renders
- **WHEN** 项目有 memberIds
- **THEN** 从 workers 数据查询对应工人信息，显示姓名、岗位、资质类型+等级+有效期
