## Context

工程管家 V1.1 原型是一个纯前端 Next.js 14 App Router 应用，数据层完全基于 `MockDataProvider`（React Context + localStorage）。没有后端 API、没有数据库、没有真正的 API 路由。

**当前技术栈**:
- Next.js 14 App Router（`src/app/(app)/` 移动端 + `src/app/pc/` PC 端）
- React Context（`MockDataProvider`）管理所有状态
- localStorage 持久化（login state + daily records）
- Tailwind CSS + 自定义 design tokens（`eng-blue`, `eng-green` 等）
- 无 API 路由目录（`src/app/api/` 不存在）

**约束**:
- 必须保持纯前端架构，不引入后端服务
- AI 功能必须在开发阶段可离线运行（Mock LLM）
- 新功能是现有表单的**并列入口**，不替代手动填表
- 数据模型扩展必须兼容现有 MockDataProvider

## Goals / Non-Goals

**Goals:**
- 在现有纯前端架构上集成 AI 能力（对话填表、智能审核、每日简报、NL 查询）
- 通过 Mock LLM 实现开发阶段零外部依赖
- 规则引擎本地执行，覆盖 80% 异常检测场景
- 新增数据模型（AiChatLog、AiAuditLog、AiBriefing）融入 MockDataProvider

**Non-Goals:**
- 不引入后端服务器或数据库
- 不替代现有 4 个手动填表页面
- 不实现 Phase 2/3 功能（进度风险预测、拍照识图）
- 不对接真实 LLM API（Phase 1 用 Mock LLM，预留接口后续切换）
- 不做语音输入（ASR）

## Decisions

### D1: AI 功能层作为 React Service 模块（非 API 路由）

**选择**: 在 `src/lib/ai/` 创建纯 TypeScript 模块，直接在客户端调用。

**原因**: 当前项目是纯前端应用，没有 API 路由。引入 `src/app/api/` 会增加架构复杂度且无法在 static export 下运行。AI 逻辑以 service 函数形式提供，React 组件直接 import 调用。

**替代方案**:
- API Routes (`/api/ai/*`): 需要 server runtime，与当前 static 架构冲突。Phase 2 切换真实 LLM 时再引入。
- Web Worker: 增加通信复杂度，Mock LLM 计算量不足以 justify。

**架构图**:
```
┌─────────────────────────────────────────────────────┐
│                   React 组件层                       │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ ChatPanel │  │ ReviewAI │  │ Briefing │          │
│  │ (移动端)  │  │ (PC端)   │  │ (PC端)   │          │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘          │
│        │             │             │                │
│  ┌─────┴─────────────┴─────────────┴──────────┐    │
│  │           src/lib/ai/ (Service 层)          │    │
│  │                                            │    │
│  │  ┌────────────┐  ┌──────────────────────┐  │    │
│  │  │ llm-client │  │ rule-engine           │  │    │
│  │  │ (Mock/Real)│  │ (异常检测6类规则)      │  │    │
│  │  └────────────┘  └──────────────────────┘  │    │
│  │  ┌────────────┐  ┌──────────────────────┐  │    │
│  │  │ prompts    │  │ context-manager       │  │    │
│  │  │ (模板管理) │  │ (项目数据注入)         │  │    │
│  │  └────────────┘  └──────────────────────┘  │    │
│  └──────────────────────┬─────────────────────┘    │
│                         │                          │
│  ┌──────────────────────┴─────────────────────┐    │
│  │         MockDataProvider (数据层)           │    │
│  │  dailyRecords + aiChatLogs + aiAuditLogs   │    │
│  │  + aiBriefings                              │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### D2: Mock LLM + LLM 接口抽象

**选择**: 定义 `LLMClient` 接口，提供 `MockLLMClient`（开发阶段）和 `ClaudeLLMClient`（生产阶段）两个实现。

**原因**: 开发阶段需要零外部依赖、零延迟、可预测的输出。Mock LLM 用规则匹配模拟意图识别和文本生成，保证开发体验。

**接口设计**:
```typescript
interface LLMClient {
    chat(messages: ChatMessage[], options?: LLMOptions): Promise<LLMResponse>;
    chatStream?(messages: ChatMessage[], options?: LLMOptions): AsyncIterable<LLMChunk>;
}

// MockLLMClient: 基于关键词匹配 + 模板生成
// ClaudeLLMClient: 调用 Claude API（Phase 2 引入 API Routes 后启用）
```

**Mock LLM 策略**:
- 意图识别：关键词匹配（"焊"→进度模块, "领"→材料模块, "签"→考勤模块）
- 实体提取：正则提取数字、位置、材料名
- 文本生成：预定义模板填充数据
- 不确定时返回 `needConfirm: true`，触发追问

### D3: 规则引擎纯 TypeScript 实现

**选择**: 在 `src/lib/ai/rule-engine.ts` 实现可配置的规则引擎。

**原因**: 6 类异常检测规则本质是数值比较和逻辑判断，不需要 ML。纯 TS 实现零延迟、可测试、可解释。

**规则引擎结构**:
```typescript
interface AuditRule {
    id: string;
    name: string;
    severity: "正常" | "疑似异常" | "明显问题";
    check(record: DailyRecord, context: AuditContext): AuditResult | null;
}

interface AuditContext {
    history: DailyRecord[];       // 近7天同模块记录
    project: Project;             // 项目信息
    attendance: DailyRecord | null; // 当日考勤
    workers: Worker[];            // 项目成员（含证件）
}
```

**6 类规则**:
1. `spike` — 数据突变：`currentQuantity > avg(history, 7) * 3`
2. `attendance_mismatch` — 考勤矛盾：`quantity > attendance.count * PRODUCTIVITY_PER_PERSON * 1.5`
3. `material_anomaly` — 材料异常：`materialQuantity > remainingDemand * 1.2`
4. `compliance_gap` — 合规缺失：高风险工序无对应安全交底记录
5. `cert_expired` — 证件过期：施工人员证件到期日 < 当前日期
6. `progress_deviation` — 进度偏离：`|actual% - planned%| > 20%`

### D4: 对话浮窗为独立 UI 组件

**选择**: 在 `src/components/ai/` 创建独立的 `AiChatWidget` 组件，通过 React Portal 挂载到页面底部。

**原因**: 浮窗需要在所有移动端页面可见，Portal 避免在每个页面重复渲染。对话状态通过 `useAiChat` hook 管理，与页面解耦。

**组件结构**:
```
src/components/ai/
├── AiChatWidget.tsx      # 浮窗容器（Portal + 拖拽）
├── AiChatBubble.tsx      # 对话气泡（用户/AI）
├── AiChatQuickActions.tsx # 快捷操作面板
└── AiBriefingCard.tsx    # 简报卡片（PC端/移动端）
```

### D5: 新增数据模型融入 MockDataProvider

**选择**: 在 MockDataProvider 的 `MockData` 接口中新增 3 个数组字段，使用与 dailyRecords 相同的 localStorage 持久化模式。

**数据模型**:
```typescript
interface AiChatLog {
    id: number;
    userId: number;
    projectId: number;
    sessionId: string;
    role: "user" | "assistant";
    content: string;
    parsedIntent?: {
        module: "考勤" | "进度" | "材料" | "施工记录";
        action: "create" | "query" | "update";
        fields: Record<string, unknown>;
    };
    createdAt: string;
}

interface AiAuditLog {
    id: number;
    dailyRecordId: number;
    result: "正常" | "疑似异常" | "明显问题";
    reason: string;
    confidence: number;
    humanDecision: "通过" | "退回" | null;
    createdAt: string;
}

interface AiBriefing {
    id: number;
    projectId: number;
    date: string;
    content: string;
    risks: Array<{
        type: string;
        severity: "高" | "中" | "低";
        description: string;
    }>;
    createdAt: string;
}
```

**扩展方式**: MockDataProvider 新增 `addAiChatLog`、`addAiAuditLog`、`addAiBriefing` 方法，初始数据从 `src/mock/ai-*.json` 加载。

### D6: 对话上下文管理

**选择**: 使用 `sessionId` 关联对话，在 `AiChatLog` 中存储完整的对话历史。

**上下文注入策略**:
- **项目数据**: 当前项目的工序列表、材料列表（从 MockDataProvider 读取）
- **历史对话**: 同一 sessionId 的所有聊天记录
- **昨日数据**: 昨日同模块的 dailyRecord，支持"跟昨天一样"语义
- **快捷操作**: 预定义的"全部帮我填"引导流程

上下文窗口限制: 最近 10 条对话 + 1 条项目数据摘要。

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Mock LLM 意图识别不够灵活 | 对话助手只能识别预定义模式 | 关键操作需用户确认；不确定时追问；Phase 2 切换真实 LLM |
| 规则引擎阈值硬编码 | 不同项目/工序的合理范围差异大 | 阈值可配置化（在规则定义中提取常量）|
| AiChatLog 数据量增长 | localStorage 5MB 限制 | 仅保留最近 30 天对话记录，定期清理 |
| 浮窗遮挡页面内容 | 影响移动端操作体验 | 可拖拽位置、可收起、底部 tab 区域自动避开 |
| PC 端审核面板改动大 | 现有审核流程中断 | 渐进式改造：先加 AI 预览卡片，不改变原操作流程 |

## Open Questions

- Mock LLM 的关键词匹配表需要根据实际使用反馈迭代（Phase 1 先用 PRD 中的示例）
- `sessionId` 的生命周期：是按天、按次打开、还是按项目？建议按次打开（组件挂载时生成 UUID）
- 审核面板的 AI 自动通过是否需要项目级别的开关？建议 Phase 1 默认不自动通过，仅标记分类
