## 1. AI 服务基础设施

- [x] 1.1 创建 `src/lib/ai/types.ts` — 定义 LLMClient 接口、ChatMessage、LLMResponse、LLMChunk 类型
- [x] 1.2 创建 `src/lib/ai/mock-llm-client.ts` — MockLLMClient 实现，关键词匹配意图识别 + 模板生成
- [x] 1.3 创建 `src/lib/ai/prompts.ts` — Prompt 模板管理（对话助手 System Prompt、简报生成 Prompt、NL 查询 Prompt），支持变量注入
- [x] 1.4 创建 `src/lib/ai/context-manager.ts` — ContextManager 模块，组装对话上下文（项目数据 + 历史对话 + 昨日记录），窗口限制 10 条

## 2. 数据模型扩展

- [x] 2.1 创建 `src/mock/ai-chat-logs.json` — AiChatLog 初始 Mock 数据
- [x] 2.2 创建 `src/mock/ai-audit-logs.json` — AiAuditLog 初始 Mock 数据
- [x] 2.3 创建 `src/mock/ai-briefings.json` — AiBriefing 初始 Mock 数据
- [x] 2.4 修改 `src/lib/MockDataProvider.tsx` — 新增 AiChatLog、AiAuditLog、AiBriefing 类型定义，扩展 MockData 接口，添加 addAiChatLog / addAiAuditLog / addAiBriefing / getAiChatLogsBySession 等方法

## 3. 规则引擎

- [x] 3.1 创建 `src/lib/ai/rule-engine.ts` — 定义 AuditRule 接口、AuditContext、AuditResult 类型
- [x] 3.2 实现 spike 规则（数据突变）— 当前完成量 > 近7天日均 × 3
- [x] 3.3 实现 attendance_mismatch 规则（考勤矛盾）— 完成量 > 出勤人数 × 人均工效 × 1.5
- [x] 3.4 实现 material_anomaly 规则（材料异常）— 领料量 > 剩余需求 × 1.2
- [x] 3.5 实现 compliance_gap 规则（合规缺失）— 高风险作业无安全交底
- [x] 3.6 实现 cert_expired 规则（证件过期）— 签到人员证件到期日 < 当前日期
- [x] 3.7 实现 progress_deviation 规则（进度偏离）— 实际进度与计划进度偏差 > 20%
- [x] 3.8 创建 `src/lib/ai/rule-engine.test.ts` — 规则引擎单元测试

## 4. AI 对话填表助手

- [x] 4.1 创建 `src/components/ai/AiChatWidget.tsx` — 浮窗容器组件（Portal 挂载、可收起、圆形按钮触发）
- [x] 4.2 创建 `src/components/ai/AiChatBubble.tsx` — 对话气泡组件（用户/AI 消息样式）
- [x] 4.3 创建 `src/components/ai/AiChatQuickActions.tsx` — 快捷操作面板（4 模块 + 全部帮我填，动态禁用已填模块）
- [x] 4.4 创建 `src/hooks/useAiChat.ts` — 对话状态管理 hook（sessionId、消息列表、发送消息、确认提交）
- [x] 4.5 修改 `src/app/(app)/layout.tsx` — 引入 AiChatWidget 组件，使浮窗在所有移动端页面可见

## 5. AI 智能审核

- [x] 5.1 创建 `src/lib/ai/smart-review.ts` — 预审流程（接收 DailyRecord，运行规则引擎，生成 AiAuditLog）
- [x] 5.2 创建 `src/components/ai/AiReviewSummary.tsx` — PC 端 AI 预审结果摘要卡片（正常/疑似异常/明显问题分类统计）
- [x] 5.3 创建 `src/components/ai/AiReviewDetail.tsx` — 异常记录详情面板（展示 AI 异常原因、建议、通过/退回操作）
- [x] 5.4 修改 `src/app/pc/review/page.tsx` — 集成 AiReviewSummary 和 AiReviewDetail 组件
- [x] 5.5 实现一键批量通过正常记录功能

## 6. AI 每日施工简报

- [x] 6.1 创建 `src/lib/ai/briefing-generator.ts` — 简报数据聚合 + Mock LLM 文本生成
- [x] 6.2 创建 `src/components/ai/AiBriefingCard.tsx` — 简报摘要卡片（PC 端 + 移动端复用）
- [x] 6.3 修改 `src/app/pc/dashboard/page.tsx` — Dashboard 页面顶部新增今日简报卡片
- [x] 6.4 修改 `src/app/(app)/home/page.tsx` — 移动端首页新增简报摘要（仅项目经理角色可见）

## 7. 自然语言数据查询

- [x] 7.1 创建 `src/lib/ai/nl-query.ts` — Text-to-Query 转换（Mock LLM 识别查询类型和参数）+ 查询执行 + 结果自然语言渲染
- [x] 7.2 创建 `src/components/ai/AiQueryPanel.tsx` — NL 查询结果面板（自然语言回答 + 关键数据卡片）
- [x] 7.3 修改 PC 端顶部布局 — 搜索栏增强为 NL 查询入口，集成 AiQueryPanel

## 8. 集成与验证

- [x] 8.1 对话助手端到端验证：浮窗 → 输入自然语言 → 意图识别 → 确认 → 创建 DailyRecord
- [x] 8.2 智能审核端到端验证：提交记录 → 规则引擎预审 → PC 端展示分类 → 经理操作
- [x] 8.3 简报端到端验证：数据聚合 → 简报生成 → PC 端/移动端展示
- [x] 8.4 NL 查询端到端验证：输入问题 → 查询转换 → 结果展示
- [x] 8.5 `npm run build` 无错误
