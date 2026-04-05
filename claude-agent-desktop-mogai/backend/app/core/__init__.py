# -*- coding: utf-8 -*-
"""
Core module for Towngas Manus Backend
核心模块 - 包含 Agent Engine、Session Manager、Skills Registry 等核心组件
"""

from .config import settings
from .agent_engine import AgentEngine, Message, ToolCall
from .session_manager import SessionManager
from .skills_registry import SkillsRegistry
from .context_manager import ContextManager
from .exceptions import (
    # MCP 异常
    MCPTOOLNotFoundError,
    MCPServerNotFoundError,
    MCPHandlerNotRegisteredError,
    MCPExecutionError,
    MCPTimeoutError,
    MCPInvalidToolNameError,
    # Agent 异常
    AgentAPIKeyError,
    AgentAPIError,
    AgentToolExecutionError,
    SessionNotFoundError,
)

__all__ = [
    # 配置
    "settings",
    # 核心组件
    "AgentEngine",
    "SessionManager",
    "SkillsRegistry",
    "ContextManager",
    # 数据类
    "Message",
    "ToolCall",
    # MCP 异常
    "MCPTOOLNotFoundError",
    "MCPServerNotFoundError",
    "MCPHandlerNotRegisteredError",
    "MCPExecutionError",
    "MCPTimeoutError",
    "MCPInvalidToolNameError",
    # Agent 异常
    "AgentAPIKeyError",
    "AgentAPIError",
    "AgentToolExecutionError",
    "SessionNotFoundError",
]
