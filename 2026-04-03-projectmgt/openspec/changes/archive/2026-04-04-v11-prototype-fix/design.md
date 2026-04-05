## Context

V1.0 原型基于 Next.js 14 App Router，使用 `MockDataProvider`（React Context）管理所有状态。移动端路由在 `(app)` group 下，PC 端在 `pc/` 目录下。所有数据存在 JSON 文件和 localStorage 中。

当前架构：
```
RootLayout → Providers(MockDataProvider) → (app)Layout → 各移动端页面
                                            → PcLayout → 各PC端页面
```

关键技术约束：
- 纯前端原型，无后端 API
- hydration 必须一致（刚修复了这个问题，不能引入新的 hydration bug）
- 所有新组件必须是 `"use client"` 组件
- MockDataProvider 是唯一数据源

## Goals / Non-Goals

**Goals:**
- 所有阻断 bug 修复（PC 项目详情 404、签到卡死、进度空数据）
- 每个页面有丰富 mock 数据展示各种状态
- 照片区域可交互（模拟拍照）
- 表单有基本验证
- 非班组长角色能正常使用

**Non-Goals:**
- 真实照片上传（不需要 file input 或相机 API）
- 真实签名功能
- 真实 GPS 定位
- 离线缓存 / PWA
- 国际化
- 单元测试（原型阶段不做）

## Decisions

### D1: 照片模拟方案 — 点击即完成

**选择**: 点击照片区域 → 弹出底部 Sheet（"拍照"/"从相册选择"）→ 点击后延迟 300ms 显示预设图片。

**理由**: 不引入真实 file input（移动端兼容性差、需要处理文件读取），用预设 picsum 图片模拟即可。用户体验接近真实，实现成本低。

**替代方案**: 用 `<input type="file" capture="camera">` — 排除，因为 PC 浏览器不支持 capture，且演示时没有真实照片可拍。

### D2: 角色适配 — 统一 getMyProject 工具函数

**选择**: 提取公共函数 `getMyProject(currentUser)`，班组长自动匹配，其他角色返回全部项目的第一个。在需要时显示项目选择器。

**理由**: 避免每个页面重复写角色判断逻辑。项目选择器只在首页展示即可，功能页面自动继承选择的项目。

**替代方案**: 在 MockDataProvider 中增加 selectedProject state — 排除，因为改动面太大，原型阶段不需要全局项目切换。

### D3: 数据扩充 — 直接编辑 JSON 文件

**选择**: 手动编写完整的 `daily-records.json`，覆盖 7 天 × 3 项目。

**理由**: 数据需要逻辑自洽（累计进度、签到人数与项目成员对应），生成脚本反而更复杂。直接写 JSON 可控性最高。

### D4: 公共组件放置位置

**选择**: 放在 `src/components/` 目录，与现有 `BottomTabBar.tsx` 同级。

**新增组件**:
- `PhotoCapture.tsx` — 模拟拍照
- `ProjectSelector.tsx` — 项目选择器
- `MaterialQuickSelect.tsx` — 材料快捷选择
- `StatusBadge.tsx` — 状态标签（PC 端已有，移动端复用）

### D5: PC 项目详情页布局

**选择**: 单页面 Tab 布局，Tab 包括：基本信息、工程量进度、近期日报、项目成员。

**理由**: 信息量大，单页面平铺会很长。Tab 切换可以按需查看。用 URL hash 不需要新建路由（`?tab=progress`），保持单文件。

**替代方案**: 多个子路由 — 排除，原型阶段过度设计。

## Risks / Trade-offs

| Risk | 影响 | 缓解措施 |
|------|------|---------|
| daily-records.json 文件过大（~52条） | 加载速度 | 52 条 JSON 约 20KB，可接受 |
| picsum.photos 图片加载慢 | 演示体验 | 用 seed 参数保证缓存命中 |
| 模拟拍照缺乏真实感 | 演示说服力 | 加时间戳水印模拟、延迟动画 |
| 非班组长角色看到第一个项目 | 可能不直观 | 首页显示项目名称，暗示当前项目 |
