# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Towngas Manus** is an enterprise AI Agent platform for Hong Kong Towngas (港华燃气), built on Claude Agent SDK. Architecture: "1 Main Agent + N Skills + M Subagents" pattern.

## Common Commands

### Development
```bash
# Start dev environment (both frontend & backend)
./scripts/dev.sh --install

# Backend only (from project root)
source venv/bin/activate && uvicorn backend.app.main:app --reload --port 8000

# Frontend only
cd frontend && npm run dev

# Type check frontend
cd frontend && npm run type-check

# Build frontend
cd frontend && npm run build
```

### Testing
```bash
# Run all backend tests
cd backend && pytest

# Run specific test file
cd backend && pytest tests/test_agent_engine.py -v

# Run single test
cd backend && pytest tests/test_skills_registry.py::test_skill_registration -v
```

### Docker
```bash
docker-compose up -d          # Start services
docker-compose logs -f        # View logs
docker-compose down           # Stop services
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Three-Layer Architecture                  │
├─────────────────────────────────────────────────────────────┤
│  Electron (desktop) → Vue 3 (frontend) → FastAPI (backend) │
│                                           ↓                  │
│                              Claude API (Anthropic SDK)     │
└─────────────────────────────────────────────────────────────┘
```

### Core Backend Modules (`backend/app/core/`)

| Module | Purpose |
|--------|---------|
| `agent_engine.py` | Claude API integration, message processing, tool orchestration |
| `skills_registry.py` | Plugin system for skills (tools callable by Claude) |
| `session_manager.py` | Conversation persistence in `backend/data/sessions/` |
| `context_manager.py` | Token estimation & context compression for long conversations |
| `config.py` | Pydantic Settings, loads from `.env` and environment variables |

### Key Data Flow

```
User → Vue ChatInterface → POST /api/chat/stream → Agent Engine → Claude API
                                                           ↓
                        SSE stream ← Skills Registry ← tool_use response
```

### Frontend Structure (`frontend/src/`)

- `components/ChatInterface.vue` - Main chat UI, renders Markdown with DOMPurify XSS protection
- `api/chat.ts` - HTTP/SSE client using fetch + ReadableStream
- `types/index.ts` - TypeScript interfaces for Session, Message, Skill

## Adding New Skills

1. Create skill file in `backend/skills/` or add to existing module
2. Register in `backend/app/core/skills_registry.py`:
```python
class MySkill(Skill):
    name = "my_skill"
    description = "Description for Claude to understand when to use this"
    parameters = {"type": "object", "properties": {...}}

    async def execute(self, **params):
        return {"result": "..."}
```
3. Skills are auto-converted to Claude tool definitions via `get_claude_tools()`

## Environment Variables

Required in `.env`:
- `ANTHROPIC_API_KEY` - Claude API key (mandatory)

Optional:
- `ANTHROPIC_MODEL` - Default: `claude-sonnet-4-20250514`
- `TOP_API_BASE_URL` / `TOP_API_KEY` - Towngas TOP system integration
- `LOG_LEVEL` - Default: `INFO`
- `CORS_ORIGINS` - Default: `["http://localhost:5173"]`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat/stream` | SSE streaming chat |
| GET/POST/DELETE | `/api/sessions` | Session CRUD |
| GET | `/api/skills` | List available skills |
| GET | `/api/health` | Health check |

## Important Patterns

- **SSE Streaming**: Backend yields `data: {"type":"text","content":"..."}\n\n`, frontend uses ReadableStream reader
- **Context Compression**: Triggered when messages > `context_compress_threshold`, preserves recent messages + summarizes older ones
- **Token Estimation**: Chinese chars / 1.5 + other chars / 4 (approximation for mixed content)
