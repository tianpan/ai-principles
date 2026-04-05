# OpenSpec 执行提示词：ai-assistant-phase1

> 将此提示词完整粘贴到新的 Claude Code 会话中执行。工作目录：`/Users/admin/Documents/00-Projects/2026-04-03-projectmgt/app`

---

## 环境变量配置

执行前先设置环境变量（智谱 GLM API，Anthropic 兼容接口）：

```bash
export ANTHROPIC_AUTH_TOKEN="f574f7a8f0c346f8b99515d37f740aac.L4Wbg55TjR47yOeb"
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
export ANTHROPIC_MODEL="glm-5.1"
```

---

## 执行提示词

```
你是一个高级全栈工程师，负责实施 OpenSpec change `ai-assistant-phase1`。

## 任务

执行 `/opsx:apply ai-assistant-phase1`，完成所有 32 个实施任务。

## 关键路径信息

**项目根目录**: `/Users/admin/Documents/00-Projects/2026-04-03-projectmgt/`
**APP 源码目录**: `/Users/admin/Documents/00-Projects/2026-04-03-projectmgt/app/`
**OpenSpec 目录**: `/Users/admin/Documents/00-Projects/2026-04-03-projectmgt/openspec/`（注意：不是 `app/openspec/`！）

**工件位置**（必须先读完这些再动手）:
- proposal: `openspec/changes/ai-assistant-phase1/proposal.md`
- design: `openspec/changes/ai-assistant-phase1/design.md`
- specs: `openspec/changes/ai-assistant-phase1/specs/**/*.md`
- tasks: `openspec/changes/ai-assistant-phase1/tasks.md`

## 当前项目架构（严格遵守）

- **Next.js 14 App Router**，纯前端，无后端
- **数据层**: `src/lib/MockDataProvider.tsx`（React Context + localStorage）
- **页面结构**: `src/app/(app)/` 移动端，`src/app/pc/` PC 端
- **组件**: `src/components/`（BottomTabBar, ProjectSelector, MaterialQuickSelect, PhotoCapture）
- **Mock 数据**: `src/mock/`（users.json, projects.json, workers.json, templates.json, daily-records.json）
- **样式**: Tailwind CSS + design tokens（eng-blue, eng-green, eng-orange, eng-gray 等）
- **无 API 路由**：AI 功能在 `src/lib/ai/` 作为客户端 Service 模块实现

## 实施规则（必须遵守）

### 1. 读取优先
每个任务开始前，必须先读取相关的设计文档和 spec 文件。特别是：
- MockDataProvider.tsx（理解现有数据层再扩展）
- 现有页面文件（理解现有 UI 模式再改造）
- Tailwind 配置（理解 design tokens 再写样式）

### 2. 代码质量
- **不可变数据**: 永远创建新对象，不修改已有对象（`{ ...prev, field: value }`）
- **文件大小**: 每个文件 < 400 行，函数 < 50 行
- **类型安全**: 所有新增代码必须有 TypeScript 类型定义
- **一致性**: 新组件风格与现有组件一致（参考 BottomTabBar.tsx 和 ProjectSelector.tsx 的模式）
- **错误处理**: 每个 AI 调用都要 try-catch，失败时降级到手动操作

### 3. AI/LLM 集成策略
采用**双模式架构**：
- **MockLLMClient**: 开发默认模式，纯规则匹配 + 模板，零外部依赖
- **RealLLMClient**: 通过环境变量切换，调用智谱 GLM API（Anthropic 兼容接口）

环境变量判断逻辑：
```typescript
const useRealLLM = typeof window !== 'undefined' && process.env.NEXT_PUBLIC_USE_REAL_LLM === 'true';
```

RealLLMClient 调用方式（Anthropic SDK 兼容）：
```typescript
// 通过 fetch 直接调用 Anthropic 兼容接口
const response = await fetch('https://open.bigmodel.cn/api/anthropic/v1/messages', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'x-api-key': 'f574f7a8f0c346f8b99515d37f740aac.L4Wbg55TjR47yOeb',
        'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
        model: 'glm-5.1',
        max_tokens: 1024,
        messages: [...],
    }),
});
```

**重要**：API Key 存储在环境变量 `NEXT_PUBLIC_GLM_API_KEY` 中，不要硬编码。在 `.env.local` 文件中配置：
```
NEXT_PUBLIC_USE_REAL_LLM=true
NEXT_PUBLIC_GLM_API_KEY=f574f7a8f0c346f8b99515d37f740aac.L4Wbg55TjR47yOeb
NEXT_PUBLIC_GLM_BASE_URL=https://open.bigmodel.cn/api/anthropic
NEXT_PUBLIC_GLM_MODEL=glm-5.1
```

### 4. 测试要求
- 规则引擎（Task 3.1-3.7）必须有完整的单元测试（Task 3.8）
- 使用现有项目的测试框架（检查是否已配置 jest/vitest，如果没有则安装 vitest）
- 测试命令必须通过：`npx vitest run`
- 规则引擎测试覆盖所有 6 条规则 + 边界情况

### 5. 任务执行顺序
严格按照 tasks.md 中的编号顺序执行（1→2→3→4→5→6→7→8），因为后面的任务依赖前面的基础设施。

每完成一个任务：
1. 在 tasks.md 中将 `- [ ]` 改为 `- [x]`
2. 运行 `npm run build` 确认无编译错误（至少每完成一组任务后验证一次）

### 6. 实施细节指南

#### Group 1: AI 服务基础设施
- `types.ts`: 导出所有 AI 相关类型（LLMClient, ChatMessage, LLMResponse, ParsedIntent 等）
- `mock-llm-client.ts`: 实现关键词匹配意图识别。关键词表：
  - 进度: "焊", "铺", "挖", "装", "米", "根", "户", "完成"
  - 材料: "领", "退", "料", "管", "设备"
  - 考勤: "签", "到", "来", "没来", "人"
  - 施工记录: "记录", "施工", "安装", "开挖"
- `prompts.ts`: 导出函数 `getChatSystemPrompt(projectContext)`, `getBriefingPrompt(data)`, `getNLQueryPrompt(question)`
- `context-manager.ts`: 从 MockDataProvider 读取项目数据，组装 messages 数组

#### Group 2: 数据模型扩展
- Mock JSON 文件结构参考现有 `daily-records.json` 的格式
- MockDataProvider 扩展必须向后兼容（现有功能不受影响）
- 新增方法签名参考现有 `addDailyRecord` 的模式

#### Group 3: 规则引擎
- 每条规则是纯函数：(record, context) => AuditResult | null
- AuditResult 包含：ruleId, severity, reason, suggestion
- 阈值作为可配置常量导出

#### Group 4: 对话填表助手
- AiChatWidget 使用 React Portal (`createPortal`)
- 浮窗按钮固定在视口右下角（`fixed bottom-20 right-4 z-50`，避免遮挡 BottomTabBar）
- 对话面板从底部滑出（`translate-y-full` → `translate-y-0` transition）
- useAiChat hook 管理状态：sessionId, messages[], loading, error

#### Group 5: 智能审核
- 不改变现有审核操作流程，只增加 AI 预审展示
- 预审结果作为摘要卡片显示在审核列表上方
- 一键批量通过需要确认对话框

#### Group 6: 每日简报
- 简报使用 MockLLMClient 生成（模板填充）
- 模板格式参考 PRD 中的简报 ASCII 示例
- BriefingCard 组件 PC/移动端复用

#### Group 7: NL 查询
- 搜索栏保持现有样式，增加 AI 图标标识
- 查询面板以浮动下拉方式展示（类似搜索建议）
- 不影响现有搜索功能

### 7. 最终验证清单
完成所有任务后，执行：
```bash
npm run build          # 编译无错误
npm run lint           # Lint 无错误
npx vitest run         # 测试全部通过
```

然后逐一验证：
- [ ] 移动端首页右下角出现 AI 浮窗按钮
- [ ] 点击浮窗展开对话面板，可收起
- [ ] 输入"今天焊了80米"能识别为进度模块
- [ ] 确认后创建 dailyRecord
- [ ] PC 端审核页面显示 AI 预审摘要
- [ ] PC 端 Dashboard 显示今日简报卡片
- [ ] PC 端搜索栏支持 NL 查询
- [ ] 所有现有功能正常（登录、填表、审核、查看项目）

## 禁止事项
- 禁止在 committed 文件中硬编码 API Key
- 禁止修改现有 4 个表单页面的核心逻辑（只允许添加浮窗入口）
- 禁止引入新的 npm 依赖（除测试框架外）
- 禁止修改 Tailwind 配置（使用现有 design tokens）
- 禁止使用 `any` 类型
- 禁止使用 `// @ts-ignore` 或 `// @ts-nocheck`
```
