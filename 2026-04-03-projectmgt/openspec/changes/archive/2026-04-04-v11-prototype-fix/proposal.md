## Why

V1.0 原型框架已搭建完成，但实际测试发现多个阻断性 bug（PC 项目详情页 404、安全签到流程卡死、进度数据不显示）和数据严重不足问题。作为演示原型，每个流程必须端到端跑通，数据丰富可展示，否则无法让业务方理解系统价值。PRD 详见 `Docs/PRD/V1.1-原型优化PRD.md`。

## What Changes

### P0 阻断修复
- 创建 PC 项目详情页 `/pc/projects/[id]/page.tsx`，展示项目信息、工程量进度、近期日报、成员列表
- 修复安全签到流程：照片区域增加模拟拍照交互，非班组长角色增加项目选择器
- 修复施工进度页面 `useState` 初始化后不跟随 hydration 更新的 bug
- 材料设备、施工记录页面的角色适配（非班组长可选择项目）

### P1 数据丰富化
- 扩充 `daily-records.json` 从 12 条到 ~52 条，覆盖 7 天 × 3 项目 × 4 模块
- 确保进度数据逻辑自洽（累计完成量不超过目标量）
- 状态分布：已通过 ~35、待审核 ~8、已退回 ~3

### P2 交互优化
- 所有照片区域增加模拟拍照交互（点击 → 延迟 → 显示 picsum 图片）
- 材料设备页增加常用材料快捷选择、领料单号自动生成
- 各表单增加验证规则和提示
- 列表页增加空状态提示

## Capabilities

### New Capabilities
- `photo-simulation`: 模拟拍照交互组件，统一照片上传/展示体验
- `project-selector`: 角色适配的项目选择器，支持班组长自动匹配和其他角色手动选择
- `material-quick-fill`: 材料设备快捷填写，包含常用材料库、自动单号、设备预设
- `pc-project-detail`: PC 端项目详情页，展示项目完整信息和数据关联

### Modified Capabilities
- `mock-data`: 扩充 daily-records 数据量，覆盖 7 天 × 3 项目，保证逻辑自洽
- `app-safety`: 修复照片交互、增加项目选择器、调整步骤验证逻辑
- `app-progress`: 修复 useState 刷新 bug、增加项目选择器
- `app-material`: 增加角色适配、常用材料选择、领料单号自动生成
- `app-site-record`: 增加角色适配、表单验证

## Impact

- **前端文件**: 新建 `pc/projects/[id]/page.tsx`；修改 `safety/page.tsx`、`progress/page.tsx`、`material/page.tsx`、`site-record/page.tsx`、`home/page.tsx`
- **Mock 数据**: `mock/daily-records.json` 大幅扩充
- **公共组件**: 可能新增 `PhotoSimulator`、`ProjectSelector`、`MaterialQuickFill` 组件
- **无后端/API 变更**: 纯前端原型，所有数据走 MockDataProvider
