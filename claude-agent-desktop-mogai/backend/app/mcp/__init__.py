# -*- coding: utf-8 -*-
"""MCP (Model Context Protocol) 集成模块。

本模块提供 MCP 工具的完整集成解决方案，包括工具的注册、执行和管理能力。
MCP 是一种标准化的工具调用协议，允许 AI 模型与外部工具进行交互。

模块结构：
    - adapter: 工具适配器，负责 MCP 工具与 Claude API 格式之间的转换
    - executor: 工具执行器，负责工具调用的实际执行、超时控制和缓存
    - registry: 工具注册表，负责工具和服务器配置的统一管理

主要组件：
    MCPToolAdapter: MCP 工具适配器，提供工具名称解析和格式转换功能。
    MCPToolDefinition: MCP 工具定义数据类，包含工具的元数据和 Schema。
    MCPToolExecutor: MCP 工具执行器，支持本地和远程执行模式。
    ToolExecutionResult: 工具执行结果数据类，包含执行状态和返回值。
    MCPToolRegistry: MCP 工具注册表，统一管理所有 MCP 工具和服务器。
    MCPServerConfig: MCP 服务器配置数据类，定义服务器的连接信息。

Example:
    基本使用流程::

        from app.mcp import MCPToolRegistry, MCPServerConfig

        # 创建注册表
        registry = MCPToolRegistry()

        # 注册服务器和工具
        registry.register_server(MCPServerConfig(name="my_server", type="local"))
        registry.register_tool("my_server", "my_tool", "工具描述", input_schema, handler)

        # 获取 Claude API 兼容的工具列表
        tools = registry.get_claude_tools()

        # 执行工具
        result = await registry.execute_tool("mcp__my_server__my_tool", {"arg": "value"})
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
