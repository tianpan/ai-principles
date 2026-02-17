# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

**IMPORTANT**: Before working on this project, read the background context in [BACKGROUND.md](./BACKGROUND.md) to understand:
- 港华集团 (Towngas Group) business context
- Project owner (田攀, VP of IT Center) and stakeholders
- AI dual-track strategy (YonAI + Lightweight AI)
- Core systems (BIP, TOP, TCIS)
- AI Runtime's positioning within the Towngas ecosystem

原始背景文件存放在 `background/` 目录下供详细查阅。

## Project Overview

AI Runtime is an AI Agent runtime platform built on Claude Agent SDK. It exposes Claude Code's agent capabilities as a service with Skills extensibility, session management, and REST API.

**定位**: AI Runtime 是 TOP 生态的智能层，与 YonAI（用友内置）互补而非竞争。

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the server
python -m ai_runtime.main
# Or with uvicorn (with hot reload)
uvicorn ai_runtime.main:app --reload

# Run all tests
pytest ai_runtime/tests

# Run a single test file
pytest ai_runtime/tests/test_session_manager.py -v

# Lint
ruff check .
ruff format .

# Frontend development
cd frontend && npm install
npm run dev           # Development server
npm run build         # Build for production

# Docker
docker-compose up --build
```

## Architecture

### Layered Structure

```
API Layer (FastAPI routes) → Core Layer (Business logic) → Storage Layer (JSON persistence)
```

### Core Layer (`ai_runtime/core/`)

**Two client patterns for agent execution:**

1. **AgentExecutor** (`agent_executor.py`) - Stateless one-shot queries using `query()` function from Claude Agent SDK
2. **AgentClient** (`agent_client.py`) - Stateful interactive sessions using `ClaudeSDKClient`, requires async context manager

Both build `ClaudeAgentOptions` with:
- `cwd`: Working directory for the agent
- `system_prompt`: System instructions
- `allowed_tools`: Default `["Read", "Write", "Bash"]`
- `permission_mode`: Set to `"acceptEdits"` for MVP

**Key components:**

- **SkillsRegistry** (`skills_registry.py`) - Decorator-based skill registration. Skills must return `{"content": [{"type": "text", "text": "..."}]}`
- **SessionManager** (`session_manager.py`) - Session lifecycle with JSON persistence. Each session gets its own workspace directory under `./data/workspaces/`
- **SubagentManager** (`subagent_manager.py`) - Manages specialized sub-agents (code-reviewer, test-writer, sql-expert, etc.)
- **CompactionManager** (`compaction_manager.py`) - Context compression for long conversations. Keeps recent N messages and summarizes older ones
- **WorkspaceManager** (`workspace_manager.py`) - Working directory management per session
- **MCPManager** (`mcp_manager.py`) - MCP Server integration

### API Layer (`ai_runtime/api/`)

FastAPI routes under `/api/v1`:
- `POST /sessions` - Create session
- `GET /sessions` - List sessions
- `GET /sessions/{id}` - Get session
- `DELETE /sessions/{id}` - Delete session
- `POST /sessions/{id}/query` - Execute query (non-streaming)
- `POST /sessions/{id}/stream` - Execute query (SSE streaming)
- `GET /sessions/{id}/messages` - Get message history
- `GET /skills` - List available skills
- `GET /skills/{name}` - Get skill details
- `GET /subagents` - List subagents
- `POST /subagents` - Register custom subagent
- `POST /subagents/{name}/execute` - Execute subagent

### Storage Layer (`ai_runtime/storage/`)

- **JSONStore** (`json_store.py`) - File-based JSON persistence with CRUD operations
- **Models** (`models.py`) - `Session` and `Message` dataclasses with serialization. `MessageRole` enum (USER, ASSISTANT, SYSTEM, TOOL)

### Configuration

All config via `ai_runtime/config.py` using pydantic-settings. Configure via `.env` file:
- `ANTHROPIC_API_KEY` - Required
- `DEFAULT_MODEL` - Default: `claude-sonnet-4-20250514`
- `MAX_TOKENS` - Default: 4096
- `MAX_TURNS` - Default: 10
- Data directories: `DATA_DIR`, `SESSIONS_DIR`, `WORKSPACES_DIR`, `SKILLS_DIR`

### Exception Hierarchy

```
AIRuntimeError (base)
├── AgentExecutionError
├── SessionError → SessionNotFoundError
├── SkillError → SkillNotFoundError, SkillExecutionError
├── StorageError
├── ConfigurationError → APIKeyMissingError
└── SubagentError
```

## Creating Skills

```python
from ai_runtime.core.skills_registry import skill

@skill(
    name="my_skill",
    description="Description",
    parameters={"input": str},
)
async def my_skill(args: dict) -> dict:
    return {"content": [{"type": "text", "text": "result"}]}
```

Place skill files in `ai_runtime/skills/builtin/` or `./data/skills/`.

## Frontend

Vue 3 + TypeScript + Vite. Uses `deep-chat` component for chat interface.
- Built output goes to `ai_runtime/static/` for FastAPI to serve
- Main component: `frontend/src/components/ChatView.vue`

## OpenSpec Workflow

This project uses OpenSpec for change management:
- `/opsx:new` - Start a new change
- `/opsx:continue` - Create next artifact
- `/opsx:apply` - Implement tasks from a change
- `/opsx:verify` - Verify implementation
- `/opsx:archive` - Archive completed change

Changes are stored in `openspec/changes/` using the spec-driven schema.
