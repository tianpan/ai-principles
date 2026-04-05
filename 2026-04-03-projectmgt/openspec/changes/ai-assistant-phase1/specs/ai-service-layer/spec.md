## ADDED Requirements

### Requirement: LLM 客户端抽象
系统 SHALL 提供 `LLMClient` 接口，支持 `chat` 方法接受消息数组并返回 AI 响应。系统 SHALL 提供 `MockLLMClient` 实现，在开发阶段使用关键词匹配 + 模板生成模拟 AI 行为。接口 SHALL 预留 `chatStream` 流式输出方法供生产环境使用。

#### Scenario: MockLLMClient 意图识别
- **WHEN** 调用 `mockLLMClient.chat([{ role: "user", content: "今天焊了80米" }])`
- **THEN** 返回结构化 JSON：`{ intent: "create", module: "进度", fields: { process: "管线焊接", quantity: 80, unit: "米" } }`

#### Scenario: MockLLMClient 文本生成
- **WHEN** 调用 `mockLLMClient.chat([{ role: "system", content: "生成简报" }, ...])`
- **THEN** 返回基于模板生成的简报 Markdown 文本

#### Scenario: 切换 LLM 实现
- **WHEN** 开发者将 LLM 客户端配置从 "mock" 切换为 "claude"
- **THEN** 所有 AI 功能使用 ClaudeLLMClient 实现
- **AND** 功能行为保持一致，仅响应质量和延迟变化

### Requirement: Prompt 模板管理
系统 SHALL 在 `src/lib/ai/prompts.ts` 中集中管理所有 Prompt 模板。每个模板 SHALL 支持 `{变量}` 占位符注入。模板 SHALL 包括：对话助手 System Prompt、简报生成 Prompt、NL 查询转换 Prompt。

#### Scenario: 对话助手 Prompt 注入项目上下文
- **WHEN** 对话助手初始化时
- **THEN** System Prompt 注入当前项目的工序列表、材料列表、班组信息
- **AND** 注入 Few-shot 示例（每种意图 3 个输入/输出对）

#### Scenario: 简报生成 Prompt 注入聚合数据
- **WHEN** 简报生成时
- **THEN** Prompt 注入当日所有项目的汇总数据
- **AND** 指定输出格式为 Markdown

### Requirement: 上下文管理器
系统 SHALL 提供 `ContextManager` 模块，负责组装 AI 请求所需的上下文数据。上下文 SHALL 包含：项目数据、历史对话、昨日记录。上下文窗口 SHALL 限制最近 10 条对话。

#### Scenario: 组装对话上下文
- **WHEN** 用户发送新消息
- **THEN** ContextManager 读取同 sessionId 的最近 10 条 AiChatLog
- **AND** 读取当前项目的工序列表和材料列表
- **AND** 读取昨日同模块的 dailyRecord（如有）
- **AND** 组装为 messages 数组传给 LLM

#### Scenario: 新会话初始化
- **WHEN** 用户首次打开对话面板
- **THEN** ContextManager 生成新 sessionId（UUID）
- **AND** 上下文仅包含项目数据和欢迎消息

### Requirement: AI 数据模型扩展
系统 SHALL 在 MockDataProvider 中新增 3 个数据集合：`aiChatLogs`、`aiAuditLogs`、`aiBriefings`。系统 SHALL 提供对应的 CRUD 方法。初始数据 SHALL 从 `src/mock/ai-*.json` 加载。

#### Scenario: 存储对话记录
- **WHEN** 用户或 AI 发送一条消息
- **THEN** 调用 `addAiChatLog` 存储为 AiChatLog
- **AND** 包含 sessionId、role、content、parsedIntent（如有）

#### Scenario: 存储审核日志
- **WHEN** AI 预审完成一条记录
- **THEN** 调用 `addAiAuditLog` 存储为 AiAuditLog
- **AND** 包含 dailyRecordId、result、reason、confidence

#### Scenario: 存储每日简报
- **WHEN** 系统生成每日简报
- **THEN** 调用 `addAiBriefing` 存储为 AiBriefing
- **AND** 包含 projectId、date、content、risks

### Requirement: 错误处理与降级
系统 SHALL 对 AI 功能提供统一的错误处理。当 AI 功能失败时，SHALL 降级到基础功能，不影响用户正常操作。

#### Scenario: LLM 调用失败
- **WHEN** MockLLMClient 处理消息时抛出异常
- **THEN** 显示"AI 暂时无法响应，请手动填表"
- **AND** 提供跳转到对应手动填表页面的链接

#### Scenario: 规则引擎计算异常
- **WHEN** 规则引擎计算时缺少必要数据（如无考勤记录）
- **THEN** 跳过依赖该数据的规则
- **AND** 继续执行其他规则，在结果中标注"部分规则未执行"
