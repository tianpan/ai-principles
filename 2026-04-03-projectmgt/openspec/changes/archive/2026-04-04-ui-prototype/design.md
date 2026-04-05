## Context

工程管家是一个面向建筑工程项目部的精细化管理 APP + PC 后台系统。PRD 已完成，定义了四大业务模块（安全考勤、进度记录、材料登记、施工记录/竣工资料）和三方角色体系（班组长-项目经理-资料员/管理层）。

本原型目标是快速搭建可交互的 UI 界面，用于业务 walkthrough 和视觉确认。不需要后端逻辑。

## Goals / Non-Goals

**Goals:**
- 覆盖 PRD 第 5 章定义的核心页面，APP 端 8 页 + PC 端 5 页
- 页面间导航可走通，表单可填写，状态可切换
- Mock 数据足够真实，能演示完整业务场景
- 可本地 `npm run dev` 一键启动

**Non-Goals:**
- 不做后端 API / 数据库
- 不做用户认证逻辑（登录页纯展示，直接跳转）
- 不做照片真实上传（用 placeholder 图片）
- 不做离线模式 / PWA
- 不做生产级部署配置

## Decisions

### 1. 技术栈: Next.js 14 + Tailwind CSS

**选择**: Next.js App Router + Tailwind CSS，不使用 Ant Design。

**理由**:
- Tailwind 纯原子化 CSS，原型阶段不需要组件库的约束，调整样式更快
- Next.js App Router 天然支持布局嵌套（APP 端底部 Tab + PC 端左侧导航）
- 减少依赖体积，安装快、启动快

**备选**: Ant Design Mobile + Ant Design — 组件丰富但定制成本高，原型阶段样式调整频繁不适合。

### 2. 路由结构: APP 端 / PC 端路径前缀分离

```
/app/login          → APP 登录页
/app/home           → APP 首页（今日待填）
/app/safety/*       → APP 安全考勤流程
/app/progress       → APP 进度记录
/app/material       → APP 材料登记
/app/site-record    → APP 施工记录
/app/review         → APP 审核列表
/app/history        → APP 历史记录
/app/profile        → APP 个人中心

/pc/login           → PC 登录页
/pc/dashboard       → PC 管理面板首页
/pc/projects/*      → PC 项目管理
/pc/review          → PC 审核工作台
/pc/reports         → PC 报表
```

**理由**: 路径前缀清晰区分两端，共享 layout 时互不干扰。APP 端 layout 限制 max-width 430px + 底部 Tab，PC 端 layout 全宽 + 左侧导航。

### 3. Mock 数据层: 独立 JSON 文件 + React Context

**结构**:
```
/src/mock/
  projects.json      ← 项目部、单项工程、工作项
  users.json         ← 用户、角色
  workers.json       ← 备案人员
  templates.json     ← 安全交底模板
  daily-records.json ← 历史每日记录（2-3天）
```

**消费方式**: 通过 React Context (`MockDataProvider`) 注入，组件从 context 读取数据。页面提交操作更新 context state（内存中），刷新重置。

**理由**: JSON 文件直观可编辑，方便业务方调整 mock 数据；Context 方式组件无感知，后续替换为 API 调用只需改 Provider。

### 4. APP 端安全考勤: 4 步向导组件

用 step wizard 模式实现（StepIndicator + 条件渲染），不引入额外向导库。签名画布用原生 Canvas API 实现。

### 5. 状态管理: React useState + Context

原型阶段不需要 Zustand/Redux。各页面状态用 useState，跨页面数据（当前用户、当前项目、mock 数据）用 Context。

### 6. 设计规范

| 项目 | 规范 |
|------|------|
| 主色 | 工程蓝 #1677ff |
| 警示色 | 安全橙 #fa8c16 |
| 成功色 | #52c41a |
| 错误色 | #ff4d4f |
| APP 正文字号 | 14px，标题 18px |
| APP 最大宽度 | 430px 居中 |
| PC 左侧导航宽 | 220px |
| 圆角 | 8px（卡片）/ 4px（按钮） |

## Skill Invocation Plan

执行各任务阶段时，显式调用以下 skills：

| 阶段 | 任务组 | 调用 Skill | 用途 |
|------|--------|-----------|------|
| 基础搭建 | 1.x Project Setup | `browse` | 脚手架完成后截图验证首页渲染 |
| 数据层 | 2.x Mock Data | `frontend-patterns` | Context+Reducer 模式实现 MockDataProvider |
| APP 页面 | 3.x-8.x APP Shell/Safety/Progress/Material/SiteRecord/Review | `design-html` | 生成高质量页面 HTML |
| APP 页面 | 3.x-8.x | `frontend-patterns` | compound component、custom hook、form handling 模式 |
| APP 页面 | 3.x-8.x | `browse` | 每完成一个页面截图验证移动端布局 |
| PC 页面 | 9.x-12.x PC Shell/Project/Review/Reports | `design-html` | 生成 PC 端页面 HTML |
| PC 页面 | 9.x-12.x | `browse` | 截图验证宽屏布局效果 |
| 集成验证 | 13.x Integration | `browse` | 端到端全流程视觉验证 |
| 集成验证 | 13.x Integration | `design-review` | 最终设计审查 |

调用方式：在执行对应任务组前，通过 `Skill` tool 调用对应 skill，按 skill 指令执行。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| 原型代码不可直接用于生产 | 明确标记 `// PROTOTYPE ONLY`，设计文档声明原型代码不作为生产基础 |
| Mock 数据结构可能与后续 API 不一致 | Mock 数据结构严格对齐 PRD 第 6 章数据模型定义 |
| Canvas 签名在不同设备表现不一 | 原型阶段仅验证交互可行性，生产需用成熟签名库 |
| Tailwind 类名导致模板可读性差 | 复杂样式抽取为 `@apply` 或独立组件 |
