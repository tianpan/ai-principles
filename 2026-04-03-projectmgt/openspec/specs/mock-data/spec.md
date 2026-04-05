## MODIFIED Requirements

### Requirement: Daily records data volume
系统 SHALL 提供足够的 daily records 数据以覆盖 3 个项目 × 7 天 × 4 个模块，总量约 52 条。

#### Scenario: Data covers 7 days for project 1
- **WHEN** 查询项目1（张三负责）的 daily records
- **THEN** 返回约 23 条记录，覆盖 03-28 至 04-03，包含考勤(7)、进度(7)、材料(5)、施工记录(4)

#### Scenario: Data covers multiple statuses
- **WHEN** 查询所有 daily records
- **THEN** 包含已通过(~35条)、待审核(~8条)、已退回(~3条) 三种状态

#### Scenario: Progress data is self-consistent
- **WHEN** 查询项目1"中压管线焊接"(id=1)的进度记录
- **THEN** 累计完成量随日期递增，每日增量合理（50-90米），总计不超过目标量1200米

### Requirement: Attendance data completeness
每条考勤记录 SHALL 包含完整的 attendance 对象：photoUrl、gpsLocation、safetyBriefingType、attendeeIds、signatureUrl。

#### Scenario: Attendance record has all fields
- **WHEN** 读取一条考勤类型的 daily record
- **THEN** attendance 对象包含非空的 photoUrl、gpsLocation、safetyBriefingType，attendeeIds 包含 3-6 个有效工人 ID
