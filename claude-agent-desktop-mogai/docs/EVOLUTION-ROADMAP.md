# Towngas Manus 演进路线图

> **文档版本**: 1.0
> **创建日期**: 2026-02-19
> **基于四位专家（架构、AI演进、前端设计、搜索）的综合分析**

---

## 一、项目愿景与对标

### 1.1 目标定位

**Towngas Manus** = 港华版 Manus

对标 C 端产品 **Manus AI**（已被 Meta 收购），打造港华集团的企业级智能 Agent 平台。

### 1.2 Manus AI 对标分析

基于搜索专家的调研发现：

| 维度 | Manus AI (C端) | Towngas Manus 目标 |
|------|---------------|-------------------|
| **核心理念** | "Less structure, more intelligence" - 行动引擎 | 企业级智能助手 - 业务赋能 |
| **技术栈** | MCP 协议 + Claude API + 多平台集成 | Claude Agent SDK + MCP + 港华系统 |
| **自主性** | 自主规划、执行、反馈 | 需增强：自主规划能力 |
| **工具生态** | 丰富的 MCP 工具（Airtable, Metabase, tl;dv） | 需扩展：港华业务系统 Skills |
| **平台支持** | Web、移动端、Slack、浏览器操作员 | Web Chat → 企微/钉钉/移动端 |

**关键发现**：
- Manus 大量使用 **MCP (Model Context Protocol)** 协议
- Manus 的 GitHub 仓库标注 "Featured on Claude!"，与 Anthropic 生态紧密集成
- 核心能力：**不仅仅是回答问题，而是执行任务**

---

## 二、当前架构评估

### 2.1 架构优势

1. **清晰的分层架构** - API 层、核心引擎层、外部服务层职责分明
2. **"1 主 Agent + N Skills" 模式** - 避免了 Agent 爆炸问题
3. **流式响应 (SSE)** - 用户体验良好
4. **上下文管理** - 自动压缩机制
5. **可扩展的技能系统** - BaseSkill 抽象类定义清晰

### 2.2 关键问题（按优先级）

| 优先级 | 问题 | 影响 | 建议解决时间 |
|--------|------|------|--------------|
| **P0** | 会话存储使用 JSON 文件 | 无法多实例部署，无事务支持 | v0.2 迁移到 PostgreSQL |
| **P1** | 缺乏企业认证机制 | 无法识别用户，无法权限控制 | v0.2 引入基础认证 |
| **P1** | 无 Subagent 支持 | 复杂场景（日报、应急处置）难以实现 | v0.3 实现 Subagent 框架 |
| **P2** | 缺乏可观测性 | 无结构化日志、无指标采集 | v0.2 引入 |
| **P2** | 无 Rate Limiting | API 无限流保护 | v0.3 |

---

## 三、架构演进路线

### 3.1 v0.2 数据接入阶段（Q2 2026）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        v0.2 架构增强                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 数据持久化升级                                                   │
│     JSON Files ──────────────────▶ PostgreSQL                      │
│     - 多实例支持                    - 数据库连接池                    │
│     - 查询索引                      - 事务支持                        │
│                                                                     │
│  2. MCP 适配器层                                                    │
│     ┌───────────┐  ┌───────────┐  ┌───────────┐                   │
│     │ TOP API   │  │ Knowledge │  │ Future    │                   │
│     │ Adapter   │  │ Adapter   │  │ Adapters  │                   │
│     └───────────┘  └───────────┘  └───────────┘                   │
│                                                                     │
│  3. 港华专属 Skills (6个 P0 场景)                                   │
│     - mcp__top__query_station (场站查询)                            │
│     - mcp__kb__search (知识检索)                                    │
│     - mcp__top__get_device_data (设备数据)                          │
│                                                                     │
│  4. RAG Pipeline                                                   │
│     文档解析 → 分块 → 向量化 → 存储 → 检索                          │
│                                                                     │
│  5. 基础认证                                                       │
│     - 支持 API Key 认证                                            │
│     - 用户识别                                                     │
│                                                                     │
│  6. 可观测性                                                       │
│     - structlog 结构化日志                                         │
│     - Prometheus 指标                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 v0.3 企业增强阶段（Q3 2026）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        v0.3 架构增强                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 企业认证集成                                                    │
│     ┌──────────────────────────────────────────┐                   │
│     │            Authentication Middleware      │                   │
│     │  ┌────────┐ ┌────────┐ ┌────────┐       │                   │
│     │  │ OAuth2 │ │ LDAP   │ │ 企微   │       │                   │
│     │  └────────┘ └────────┘ └────────┘       │                   │
│     └──────────────────────────────────────────┘                   │
│                                                                     │
│  2. Subagent 框架                                                   │
│     ┌──────────────────────────────────────────┐                   │
│     │           SubagentManager                 │                   │
│     │  - DailyReportAgent (日报生成)            │                   │
│     │  - EmergencyAgent (应急处置)              │                   │
│     │  - AnalysisAgent (数据分析)               │                   │
│     └──────────────────────────────────────────┘                   │
│                                                                     │
│  3. 消息推送                                                        │
│     - 企业微信 Webhook                                              │
│     - 钉钉机器人                                                    │
│                                                                     │
│  4. 多模态支持                                                      │
│     - 文件上传                                                      │
│     - 图片处理                                                      │
│                                                                     │
│  5. 长期记忆系统                                                    │
│     - 用户偏好存储                                                  │
│     - 跨会话记忆                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 v1.0 Agent OS 阶段（Q4 2026）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        v1.0 Agent OS 架构                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    多租户层                                   │   │
│  │  TenantContext ─▶ 租户隔离的数据访问、配置、Skills            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    权限层 (RBAC)                              │   │
│  │  Role: admin | manager | operator | viewer                   │   │
│  │  Permission: skills:* | sessions:* | reports:*              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    审计层                                     │   │
│  │  AuditLog: who | what | when | result | details             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    编排层                                     │   │
│  │  Workflow: Agent1 ─▶ Agent2 ─▶ Agent3                       │   │
│  │  State Machine: 定义任务状态转换                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Agent 市场                                 │   │
│  │  - Skills 发布与订阅                                         │   │
│  │  - Subagent 模板共享                                         │   │
│  │  - 版本管理                                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、与 Manus 的差距分析与追赶策略

### 4.1 功能差距

| 能力维度 | Manus (C端) | Towngas Manus (当前) | 追赶策略 |
|----------|-------------|----------------------|----------|
| **自主性** | 自主规划、执行、反馈 | 基础对话 + 工具调用 | v0.3: 增强 plan_and_execute 能力 |
| **记忆** | 长期记忆、用户偏好 | 会话级记忆 | v0.3: 实现 MemoryStore |
| **多模态** | 图像、文件处理 | 纯文本 | v0.3: 文件上传支持 |
| **工具生态** | 丰富的 MCP 工具 | 4 个基础 Skill | v0.2: 快速扩展到 10+ Skills |
| **协作** | 多 Agent 协作 | 单 Agent | v0.3: Subagent 框架 |
| **个性化** | 用户偏好学习 | 无 | v0.3: 用户画像系统 |
| **浏览器操作** | Manus 浏览器操作员 | 无 | v1.0: 评估是否需要 |

### 4.2 追赶路线图

```
2026 Q2 (v0.2) ──────────────────────────────────────────────────────
  ✅ 扩展工具生态到 10+ Skills
  ✅ 实现知识库 RAG
  ✅ PostgreSQL 存储
  ✅ 基础认证

2026 Q3 (v0.3) ──────────────────────────────────────────────────────
  ✅ Subagent 框架
  ✅ 长期记忆系统
  ✅ 自主规划能力 (plan_and_execute)
  ✅ 文件上传支持
  ✅ 企业微信/钉钉推送

2026 Q4 (v1.0) ──────────────────────────────────────────────────────
  ✅ 多租户支持
  ✅ RBAC 权限
  ✅ Agent 编排
  ✅ Agent 市场
```

---

## 五、首批 P0 场景（v0.2 必须实现）

基于 PRD 分析，以下 6 个场景条件成熟、价值极高，应在 v0.2 阶段优先实现：

| 排名 | 场景 | 领域 | 技术方案 | 业务价值 |
|------|------|------|----------|----------|
| 1 | **操作规程智能问答** | 场站 | RAG + 知识库 | DM50 等文档已有，解决一线员工查询难题 |
| 2 | **常见问题智能问答** | 客服 | RAG + 知识库 | FAQ 文档已有，可大幅减少人工咨询 |
| 3 | **场站运行日报自动生成** | 场站 | TOP 数据 + 模板生成 | 每日节省大量人工 |
| 4 | **运营日报自动生成** | 跨域 | 多系统数据汇总 | 展示 AI 能力的好案例 |
| 5 | **应急处置辅助** | 场站 | 知识库 + 规则引擎 | 显著提升应急响应效率 |
| 6 | **气量智能预测** | 气源 | 时序预测模型 | 直接降低采购成本 |

**共同特点**：
- 数据/知识库已存在或易获取
- 技术方案成熟（RAG、模板生成、规则引擎）
- 业务价值明确，ROI 可量化
- 可在 2-3 个月内交付

---

## 六、技术栈建议

### 6.1 当前技术栈

| 层级 | 技术 |
|------|------|
| AI 引擎 | Claude Agent SDK (Anthropic) |
| 后端框架 | FastAPI (Python) |
| 前端 | Vue 3 |
| 存储 | JSON 文件 |
| 部署 | 待定 |

### 6.2 推荐技术增强

| 领域 | 当前 | 建议 | 引入时间 |
|------|------|------|----------|
| 数据库 | JSON 文件 | PostgreSQL + SQLAlchemy | v0.2 |
| 缓存 | 无 | Redis (会话缓存) | v0.2 |
| 向量存储 | 无 | pgvector 或 Qdrant | v0.2 |
| 日志 | print | structlog + ELK/Loki | v0.2 |
| 指标 | 无 | Prometheus + Grafana | v0.2 |
| 追踪 | 无 | OpenTelemetry + Jaeger | v0.3 |
| 任务队列 | 无 | ARQ (日报生成等异步任务) | v0.3 |

---

## 七、关键代码改进建议

### 7.1 增强自主规划能力

```python
# 建议在 AgentEngine 中增加规划能力
class AgentEngine:

    async def plan_and_execute(
        self,
        user_request: str,
        context: Dict[str, Any]
    ) -> AsyncGenerator[Dict, None]:
        """
        自主规划执行流程：
        1. 分析请求，生成执行计划
        2. 逐步执行计划中的步骤
        3. 根据执行结果调整计划
        4. 汇总结果并返回
        """
        # Step 1: 规划
        plan = await self._generate_plan(user_request, context)
        yield {"type": "plan", "steps": plan.steps}

        # Step 2: 执行
        results = []
        for step in plan.steps:
            yield {"type": "step_start", "step": step.name}
            result = await self._execute_step(step)
            results.append(result)
            yield {"type": "step_result", "step": step.name, "result": result}

            # Step 3: 动态调整
            if result.needs_replan:
                plan = await self._adjust_plan(plan, result)

        # Step 4: 汇总
        summary = await self._summarize_results(results)
        yield {"type": "summary", "content": summary}
```

### 7.2 长期记忆系统

```python
# backend/app/core/memory.py
class MemoryStore:
    """长期记忆存储"""

    async def store(self, user_id: str, memory: MemoryEntry) -> None:
        """存储记忆 - 使用向量数据库，支持语义检索"""
        ...

    async def recall(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> List[MemoryEntry]:
        """召回相关记忆"""
        ...

@dataclass
class MemoryEntry:
    content: str
    memory_type: str  # preference | fact | interaction
    importance: float  # 0-1
    created_at: datetime
    metadata: Dict[str, Any]
```

### 7.3 Subagent 框架

```python
# backend/app/core/subagent_manager.py
@dataclass
class SubagentDefinition:
    name: str
    description: str
    tools: List[str]  # 可用的 Skill 名称
    system_prompt: str
    model: str = "claude-sonnet-4-5"

class SubagentManager:
    """子代理管理器"""

    def __init__(self, skills_registry: SkillsRegistry):
        self.skills_registry = skills_registry
        self._subagents: Dict[str, SubagentDefinition] = {}

    async def execute(
        self,
        subagent_name: str,
        task: str,
        context: Dict[str, Any]
    ) -> AsyncGenerator[Dict, None]:
        """执行子代理任务"""
        subagent = self._subagents[subagent_name]
        engine = AgentEngine(
            system_prompt=subagent.system_prompt,
            skills_registry=self._get_filtered_registry(subagent.tools)
        )
        async for chunk in engine.chat(task, []):
            yield chunk
```

---

## 八、数据库 Schema 建议

```sql
-- 租户表
CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- 用户表
CREATE TABLE users (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    role VARCHAR(50) NOT NULL,
    preferences JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, username)
);

-- 会话表
CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    title VARCHAR(500),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_sessions_tenant_user ON sessions(tenant_id, user_id);

-- 消息表
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);
CREATE INDEX idx_messages_session ON messages(session_id);

-- 审计日志表 (v1.0)
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    status VARCHAR(50) NOT NULL,
    details JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_audit_tenant_time ON audit_logs(tenant_id, timestamp);
```

---

## 九、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| PostgreSQL 迁移复杂 | 中 | 高 | 提前准备迁移脚本，充分测试 |
| TOP API 不稳定 | 中 | 高 | 实现熔断、降级、Mock 模式 |
| Subagent 实现复杂 | 中 | 中 | 参考 Claude Agent SDK Demo |
| 企业认证对接困难 | 中 | 中 | 提前与 IT 部门沟通接口规范 |
| 性能瓶颈 | 低 | 高 | 引入 Redis 缓存、连接池 |
| 与 YonAI 定位冲突 | 中 | 高 | 明确边界：YonAI 做 BIP，我们做所有自研系统 |

---

## 十、成功指标

### 10.1 v0.2 验收标准

- [ ] 可以查询 TOP 系统场站数据
- [ ] 可以基于知识库回答问题
- [ ] 响应准确率 > 80%
- [ ] 用户满意度 > 4.0/5.0
- [ ] PostgreSQL 存储迁移完成
- [ ] 基础认证可用

### 10.2 业务价值指标

| 指标 | v0.2 目标 | v1.0 目标 |
|------|-----------|-----------|
| 用户采用率 | 3 个子公司试点 | 全集团推广 |
| 日常活跃用户 | 50+ | 500+ |
| 平均查询响应时间 | < 5s | < 3s |
| 替代人工查询比例 | > 30% | > 60% |
| 运营日报自动化率 | > 80% | > 95% |

---

## 十一、总结

### 核心方向

Towngas Manus 的演进目标是成为 **"港华版 Manus"** - 一个企业级的智能 Agent 平台。关键路径：

1. **v0.2 (Q2 2026)**: 夯实基础
   - PostgreSQL 存储
   - 知识库 RAG
   - 6 个 P0 场景落地
   - 基础认证

2. **v0.3 (Q3 2026)**: 增强 AI 能力
   - Subagent 框架
   - 长期记忆
   - 自主规划
   - 多模态支持

3. **v1.0 (Q4 2026)**: 企业级 Agent OS
   - 多租户
   - RBAC 权限
   - 审计日志
   - Agent 市场

### 关键成功因素

1. **保持架构简洁** - 不要过度设计，按需演进
2. **快速交付价值** - 每个 Sprint 都应有可用的功能
3. **重视可观测性** - 从第一天起就要考虑监控和调试
4. **安全不是事后补救** - 从 v0.2 开始就要规划认证和审计
5. **用户反馈驱动** - 尽早让业务用户试用，收集反馈

---

**文档生成者**: Claude (综合四位专家分析)
**审核状态**: 待用户确认
