## Why

班组长每天手工填写 4 个模块的表单（安全签到、施工进度、材料设备、施工记录），耗时约 15 分钟，易出错且重复劳动。项目经理逐条审核 25+ 条日报，耗时约 20 分钟，且无法提前发现风险。当前数据流是单向的：填表 → 审核 → 存档，数据价值未被二次利用。

AI 助手的目标：**从"填表机器"变成"智能管家"** — 对话式录入降低填表成本，规则引擎+LLM 提升审核效率，自动简报释放数据价值。

## What Changes

- **新增 AI 对话填表助手**（P0）— 移动端底部浮窗组件，班组长通过自然语言描述完成登记，LLM 做意图识别+实体提取，自动调用现有 addDailyRecord 流程
- **新增 AI 智能审核**（P0）— PC 端审核面板改造，6 类异常规则引擎自动预审（数据突变/逻辑矛盾/材料异常/合规缺失/证件过期/进度偏离），异常记录高亮展示，正常记录一键批量通过
- **新增 AI 每日施工简报**（P1）— 每日 18:00 自动聚合各项目数据，LLM 生成结构化 Markdown 简报，包含全局概览、项目详情、风险提示、AI 建议
- **新增自然语言数据查询**（P1）— PC 端顶部搜索栏增强，经理用口语提问（如"XX路还剩多少管没焊？"），LLM 转为结构化查询参数，调用现有数据方法，结果以自然语言+关键数字返回
- **新增 AI 服务层** — API 路由 `/api/ai/*`，封装 LLM 调用、Prompt 管理、上下文管理、流式输出
- **新增数据模型** — `AiChatLog`（对话记录）、`AiAuditLog`（审核日志）、`AiBriefing`（每日简报）

不改动的部分：
- 现有 MockDataProvider 数据层和 localStorage 持久化保持不变
- 现有 4 个表单页面保持不变（对话助手是并列入口，不替代手动填表）
- Phase 2/3 功能（进度风险预测、拍照识图）不在本次范围

## Capabilities

### New Capabilities

- `ai-chat-assistant`: AI 对话填表助手 — 对话交互组件、意图识别、实体提取、表单自动填充、上下文管理（历史对话、昨日数据复制）
- `ai-smart-review`: AI 智能审核 — 6 类异常规则引擎、AI 预审分类（正常/疑似异常/明显问题）、审核面板改造、审核反馈闭环
- `ai-daily-briefing`: AI 每日施工简报 — 数据聚合、LLM 文本生成、简报模板渲染、多渠道分发（APP 内/PC 端）
- `ai-nl-query`: 自然语言数据查询 — Text-to-Query 转换、查询执行、结果自然语言渲染
- `ai-service-layer`: AI 服务基础设施 — LLM API 封装、Prompt 模板管理、流式输出、错误处理、Mock LLM（开发阶段）

### Modified Capabilities

（无现有 spec 需要修改）

## Impact

**前端代码**:
- `src/app/(app)/` 所有页面：新增右下角 AI 浮窗按钮（共用组件）
- `src/app/pc/review/`：审核面板 UI 改造，增加 AI 预审结果展示
- `src/app/pc/dashboard/`：新增每日简报卡片
- `src/app/pc/` 顶部布局：搜索栏增强为 NL 查询入口

**新增文件**:
- `src/components/ai/` — 对话浮窗、消息气泡、快捷操作面板
- `src/lib/ai/` — LLM 客户端、Prompt 模板、规则引擎、上下文管理
- `src/app/api/ai/` — chat、audit、briefing、query API 路由
- `src/mock/ai-*.json` — Mock AI 数据

**依赖**:
- 新增 `ai` 字段的 Mock 数据扩展（不影响现有结构）
- 开发阶段使用 Mock LLM（不依赖外部 API），生产环境切换到 Claude API

**数据模型**:
- `AiChatLog`、`AiAuditLog`、`AiBriefing` 新增到 MockDataProvider

**风险**:
- LLM 响应延迟 >3 秒影响体验 → 流式输出 + 规则引擎本地执行
- 意图识别准确率 <90% → 关键操作需用户确认
- API 成本 → Mock LLM 开发，Sonnet 级模型控制 token
