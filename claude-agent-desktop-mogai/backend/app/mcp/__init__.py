# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) 集成模块

提供 MCP 工具的注册、执行和管理能力
"""

from .adapter import MCPToolAdapter, MCPToolDefinition
from .executor import MCPToolExecutor, ToolExecutionResult
from .registry import MCPToolRegistry, MCPServerConfig

__all__ = [
    "MCPToolAdapter",
    "MCPToolDefinition",
    "MCPToolExecutor",
    "ToolExecutionResult",
    "MCPToolRegistry",
    "MCPServerConfig",
]
