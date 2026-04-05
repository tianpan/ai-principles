# CLAUDE.md

## Goal
**Towngas Manus** - 构建Towngas企业级 AI Agent 平台，基于 Claude Agent SDK。
Towngas Manus = 一个让企业在复杂世界中，通过 Agent 持续感知、决策、行动并自我优化的操作系统。


## CORE PHILOSOPHY（核心理念）

### Agent Loop
```
Goal → Action → Observation → Adjustment → Goal ...
```
Agent 的能力来自**持续循环**，而不是单次推理。

### 三大系统架构
| 系统 | 职责 | 边界 |
|------|------|------|
| **Execution** | Agent 运行、状态推进、能力调用 | 不承载业务知识 |
| **Capability** | Skills、Memory、MCP 接口 | 不控制流程 |
| **Control** | 协作编排、权限审计 | 不执行具体任务 |

> **一句话**: Towngas Manus = 用 Agent 的闭环能力，把企业"做事的方式"变成系统能力。

## HOW TO WORK（工作方式）

### 修改 UI 主题
- **CSS 变量**: `frontend/src/styles/main.css` 的 `:root` 块
- **核心变量**: `--primary-color: #3498db` (蓝色系)
- **Element Plus 覆盖**: `--el-color-primary` 等变量
