## ADDED Requirements

### Requirement: 自然语言查询入口
系统 SHALL 在 PC 端顶部搜索栏提供自然语言查询功能。用户输入自然语言问题后，系统 SHALL 将问题转为结构化查询参数，执行查询，将结果以自然语言 + 关键数字返回。

#### Scenario: 经理在搜索栏输入问题
- **WHEN** 项目经理在 PC 端顶部搜索栏输入"XX路还剩多少管没焊？"
- **THEN** 系统识别查询意图为"项目进度查询"
- **AND** 提取参数：项目=XX路, 工序=管线焊接, 查询类型=剩余量

#### Scenario: 查询结果以自然语言返回
- **WHEN** 系统完成查询
- **THEN** 在搜索栏下方弹出结果面板
- **AND** 显示自然语言回答："XX路中压管道工程，中压管线焊接，目标 1,200 米，累计完成 500 米，剩余 700 米"
- **AND** 高亮关键数字（700 米）

### Requirement: Text-to-Query 转换
系统 SHALL 将自然语言转为结构化查询参数，支持以下查询类型：项目进度查询、材料汇总查询、人员考勤查询、风险排序查询。当查询意图不明确时，系统 SHALL 追问澄清。

#### Scenario: 项目进度查询
- **WHEN** 用户输入"XX路焊了多少了"
- **THEN** 转换为：{ projectId: 1, moduleType: "进度", workItemName: "管线焊接", queryType: "cumulative" }
- **AND** 调用 `getCumulativeProgress` 获取结果

#### Scenario: 材料汇总查询
- **WHEN** 用户输入"本月材料花了多少"
- **THEN** 转换为：{ moduleType: "材料", dateRange: "本月", queryType: "summary" }
- **AND** 从 dailyRecords 聚合材料数据

#### Scenario: 人员考勤查询
- **WHEN** 用户输入"张三今天填了吗"
- **THEN** 转换为：{ teamLeaderId: 2, date: "today", queryType: "submission_status" }
- **AND** 返回已填模块列表和未填模块

#### Scenario: 风险排序查询
- **WHEN** 用户输入"三个项目哪个最可能延期"
- **THEN** 转换为：{ queryType: "risk_ranking" }
- **AND** 对所有项目计算延期概率并排序

#### Scenario: 查询意图不明确
- **WHEN** 用户输入"那个怎么样了"
- **THEN** 系统追问"您想查询哪个项目的什么情况？"

### Requirement: 查询结果展示
系统 SHALL 将查询结果以自然语言回答 + 关键数据卡片的形式展示。结果面板 SHALL 支持点击展开详情。

#### Scenario: 展示进度查询结果
- **WHEN** 查询返回进度数据
- **THEN** 结果面板显示：
  - 自然语言描述（"XX路管线焊接已完成 500/1200 米，进度 41.7%"）
  - 数据卡片：目标量、完成量、剩余量、进度百分比
  - 预计完成时间

#### Scenario: 展示材料汇总结果
- **WHEN** 查询返回材料汇总数据
- **THEN** 结果面板显示：
  - 自然语言描述（"本月全项目材料消耗汇总"）
  - 按材料类型分类的汇总表格

#### Scenario: 展示风险排序结果
- **WHEN** 查询返回风险排序
- **THEN** 结果面板显示：
  - 按风险从高到低排列的项目列表
  - 每个项目标注延期概率和预计超期天数
