# -*- coding: utf-8 -*-
"""MCP Tool Registry - MCP 工具注册表。

本模块提供 MCP 工具和服务器的统一管理功能，包括注册、发现和执行。
它是 MCP 集成的核心组件，协调 Adapter 和 Executor 的工作。

主要功能：
    - MCP Server 配置和生命周期管理
    - MCP Tool 注册、发现和管理
    - 提供与 Claude API 兼容的工具列表
    - 工具执行代理

主要组件：
    MCPServerConfig: MCP 服务器配置数据类。
    MCPToolRegistry: MCP 工具注册表类。

Example:
    完整使用流程::

        from app.mcp.registry import MCPToolRegistry, MCPServerConfig

        # 创建注册表
        registry = MCPToolRegistry()

        # 注册服务器
        registry.register_server(MCPServerConfig(
            name="top",
            type="local",
            description="港华运营平台"
        ))

        # 注册工具
        async def query_handler(station_id: str) -> dict:
            return {"id": station_id}

        registry.register_tool(
            server_name="top",
            tool_name="query_station",
            description="查询场站信息",
            input_schema={"type": "object", "properties": {...}},
            handler=query_handler
        )

        # 获取 Claude API 格式的工具列表
        tools = registry.get_claude_tools()

        # 执行工具
        result = await registry.execute_tool(
            "mcp__top__query_station",
            {"station_id": "ST001"}
        )

        # 获取统计信息
        stats = registry.get_stats()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .adapter import MCPToolAdapter, MCPToolDefinition
from .executor import MCPToolExecutor, ToolExecutionResult

# 配置模块级日志
logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """MCP Server 配置数据类。

    存储单个 MCP Server 的完整配置信息，支持本地、HTTP 和 stdio
    三种服务器类型。

    服务器类型：
        - local: 本地服务器，处理器通过代码直接注册。
        - http: HTTP 服务器，通过 URL 调用远程 MCP Server。
        - stdio: stdio 服务器，通过命令行启动子进程通信。

    Attributes:
        name: 服务器唯一标识名称，用于工具命名和查找。
        type: 服务器类型，可选值为 "local"、"http" 或 "stdio"。
        description: 服务器功能描述。
        enabled: 是否启用该服务器，禁用的服务器不会提供工具。
        url: HTTP 服务器的基础 URL（仅 type=http 时使用）。
        command: stdio 服务器的启动命令（仅 type=stdio 时使用）。
        env: 环境变量字典，传递给 stdio 服务器进程。
        metadata: 额外元数据字典，可存储自定义配置。

    Example:
        本地服务器配置::

            config = MCPServerConfig(
                name="top",
                type="local",
                description="港华运营平台",
                enabled=True
            )

        HTTP 服务器配置::

            config = MCPServerConfig(
                name="remote",
                type="http",
                url="https://api.example.com/mcp",
                description="远程 MCP 服务"
            )

        stdio 服务器配置::

            config = MCPServerConfig(
                name="external",
                type="stdio",
                command="python -m my_mcp_server",
                env={"API_KEY": "xxx"}
            )
    """

    name: str
    type: str = "local"
    description: str = ""
    enabled: bool = True
    url: str | None = None
    command: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """将配置转换为字典格式。

        Returns:
            包含所有配置属性的字典。

        Example:
            >>> config.to_dict()
            {'name': 'top', 'type': 'local', 'description': '...', ...}
        """
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "enabled": self.enabled,
            "url": self.url,
            "command": self.command,
            "env": self.env,
            "metadata": self.metadata,
        }


class MCPToolRegistry:
    """MCP 工具注册表。

    统一管理所有 MCP Server 和 Tool，提供完整的注册、发现和执行功能。
    是 MCP 集成的核心协调器，连接 Adapter 和 Executor。

    主要职责：
        - 服务器配置和生命周期管理
        - 工具注册、查询和注销
        - 生成 Claude API 兼容的工具列表
        - 代理工具执行请求

    Attributes:
        _tools: 工具定义字典，键为 MCP 格式名称。
        _servers: 服务器配置字典，键为服务器名称。
        _executor: 工具执行器实例。
        _initialized: 初始化状态标志。

    Example:
        完整使用流程::

            # 创建注册表
            registry = MCPToolRegistry()

            # 注册服务器
            registry.register_server(MCPServerConfig(
                name="top",
                type="local",
                description="港华运营平台"
            ))

            # 注册工具
            async def query_handler(station_id: str) -> dict:
                return {"id": station_id}

            registry.register_tool(
                server_name="top",
                tool_name="query_station",
                description="查询场站信息",
                input_schema={"type": "object", "properties": {...}},
                handler=query_handler
            )

            # 获取 Claude API 格式的工具列表
            tools = registry.get_claude_tools()

            # 执行工具
            result = await registry.execute_tool(
                "mcp__top__query_station",
                {"station_id": "ST001"}
            )

            # 获取统计信息
            stats = registry.get_stats()
    """

    def __init__(self, executor: MCPToolExecutor | None = None) -> None:
        """初始化 MCP 工具注册表。

        Args:
            executor: 工具执行器实例。如果为 None，将创建新的
                MCPToolExecutor 实例。传入自定义执行器可用于共享
                执行器配置或进行测试。

        Example:
            使用默认执行器::

                registry = MCPToolRegistry()

            使用自定义执行器::

                executor = MCPToolExecutor(timeout_seconds=60.0)
                registry = MCPToolRegistry(executor=executor)
        """
        self._tools: dict[str, MCPToolDefinition] = {}
        self._servers: dict[str, MCPServerConfig] = {}
        self._executor = executor or MCPToolExecutor()
        self._initialized = False

        logger.debug("MCPToolRegistry initialized")

    # ==================== 服务器管理 ====================

    def register_server(self, config: MCPServerConfig) -> None:
        """注册 MCP Server。

        将服务器配置添加到注册表，如果同名服务器已存在则覆盖。

        Args:
            config: 服务器配置对象，包含名称、类型、描述等信息。

        Example:
            >>> registry.register_server(MCPServerConfig(
            ...     name="top",
            ...     type="local",
            ...     description="港华运营平台"
            ... ))
        """
        self._servers[config.name] = config
        logger.info(
            "Registered MCP server: name=%s, type=%s, enabled=%s",
            config.name,
            config.type,
            config.enabled,
        )

    def unregister_server(self, name: str) -> bool:
        """注销 MCP Server。

        移除服务器配置，同时自动移除该服务器的所有工具和处理器。

        Args:
            name: 要注销的服务器名称。

        Returns:
            如果服务器存在且成功注销返回 True，
            如果服务器不存在返回 False。

        Example:
            >>> registry.unregister_server("top")
            True
        """
        if name not in self._servers:
            logger.warning("Attempted to unregister non-existent server: %s", name)
            return False

        # 移除服务器
        del self._servers[name]

        # 移除该服务器的所有工具
        tools_to_remove = [
            mcp_name
            for mcp_name, tool in self._tools.items()
            if tool.server_name == name
        ]
        for mcp_name in tools_to_remove:
            del self._tools[mcp_name]

        logger.info(
            "Unregistered MCP server: name=%s, tools_removed=%d",
            name,
            len(tools_to_remove),
        )
        return True

    def get_server(self, name: str) -> MCPServerConfig | None:
        """获取指定名称的服务器配置。

        Args:
            name: 服务器名称。

        Returns:
            服务器配置对象，如果不存在返回 None。
        """
        return self._servers.get(name)

    def get_all_servers(self) -> list[MCPServerConfig]:
        """获取所有服务器配置。

        Returns:
            所有服务器配置对象的列表，包括已禁用的服务器。
        """
        return list(self._servers.values())

    def get_enabled_servers(self) -> list[MCPServerConfig]:
        """获取所有启用的服务器配置。

        Returns:
            所有 enabled=True 的服务器配置列表。
        """
        return [s for s in self._servers.values() if s.enabled]

    # ==================== 工具管理 ====================

    def register_tool(
        self,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Awaitable[Any]] | None = None,
        annotations: dict[str, Any] | None = None,
    ) -> MCPToolDefinition:
        """注册 MCP 工具。

        创建工具定义并添加到注册表，同时可选地注册处理函数。

        Args:
            server_name: MCP Server 名称，用于组织工具。
            tool_name: 工具名称，在服务器内应唯一。
            description: 工具功能描述，将展示给 AI 模型。
            input_schema: 输入参数的 JSON Schema 定义。
            handler: 可选的异步处理函数。如果提供，将注册到执行器
                用于本地执行。对于远程工具（http/stdio），可为 None。
            annotations: 可选的工具注解字典，可包含权限、分类等元数据。

        Returns:
            创建的 MCPToolDefinition 实例。

        Example:
            >>> async def query_station(station_id: str) -> dict:
            ...     return {"id": station_id}
            >>> tool_def = registry.register_tool(
            ...     server_name="top",
            ...     tool_name="query_station",
            ...     description="查询场站信息",
            ...     input_schema={
            ...         "type": "object",
            ...         "properties": {
            ...             "station_id": {"type": "string"}
            ...         }
            ...     },
            ...     handler=query_station
            ... )
            >>> tool_def.mcp_name
            'mcp__top__query_station'
        """
        # 创建工具定义
        tool_def = MCPToolAdapter.create_tool_definition(
            server_name=server_name,
            tool_name=tool_name,
            description=description,
            input_schema=input_schema,
            annotations=annotations,
        )

        # 注册到工具表
        self._tools[tool_def.mcp_name] = tool_def

        # 注册处理器
        if handler is not None:
            self._executor.register_local_handler(server_name, tool_name, handler)

        logger.info(
            "Registered MCP tool: mcp_name=%s, server=%s, has_handler=%s",
            tool_def.mcp_name,
            server_name,
            handler is not None,
        )
        return tool_def

    def unregister_tool(self, mcp_name: str) -> bool:
        """注销 MCP 工具。

        从注册表移除工具定义，同时移除关联的处理器。

        Args:
            mcp_name: MCP 格式的工具名称（如 "mcp__top__query_station"）。

        Returns:
            如果工具存在且成功注销返回 True，
            如果工具不存在返回 False。

        Example:
            >>> registry.unregister_tool("mcp__top__query_station")
            True
        """
        if mcp_name not in self._tools:
            logger.warning("Attempted to unregister non-existent tool: %s", mcp_name)
            return False

        tool = self._tools[mcp_name]
        del self._tools[mcp_name]

        # 移除处理器
        self._executor.unregister_local_handler(tool.server_name, tool.name)

        logger.info("Unregistered MCP tool: mcp_name=%s", mcp_name)
        return True

    def get_tool(self, mcp_name: str) -> MCPToolDefinition | None:
        """获取指定名称的工具定义。

        Args:
            mcp_name: MCP 格式的工具名称（如 "mcp__top__query_station"）。

        Returns:
            工具定义对象，如果不存在返回 None。
        """
        return self._tools.get(mcp_name)

    def get_all_tools(self) -> list[MCPToolDefinition]:
        """获取所有工具定义。

        Returns:
            所有工具定义对象的列表。
        """
        return list(self._tools.values())

    def get_tools_by_server(self, server_name: str) -> list[MCPToolDefinition]:
        """获取指定服务器的所有工具定义。

        Args:
            server_name: 服务器名称。

        Returns:
            该服务器下的所有工具定义列表，如果服务器不存在返回空列表。
        """
        return [
            tool for tool in self._tools.values() if tool.server_name == server_name
        ]

    # ==================== Claude API 兼容 ====================

    def get_claude_tools(self) -> list[dict[str, Any]]:
        """获取 Claude API 格式的工具列表。

        将所有注册的工具转换为 Claude API messages 接口兼容的格式，
        可直接用于 API 调用时的 tools 参数。

        Returns:
            Claude API 兼容的工具定义列表，每个元素包含 name、description、
            input_schema 等字段。

        Example:
            >>> tools = registry.get_claude_tools()
            >>> # 可直接用于 Claude API
            >>> response = client.messages.create(
            ...     model="claude-3-opus-20240229",
            ...     messages=[...],
            ...     tools=tools
            ... )
        """
        return [MCPToolAdapter.to_claude_tool(tool) for tool in self._tools.values()]

    def get_tool_names(self) -> list[str]:
        """获取所有工具的 MCP 格式名称列表。

        Returns:
            所有工具名称的列表，格式为 "mcp__{server}__{tool}"。
        """
        return list(self._tools.keys())

    # ==================== 工具执行 ====================

    async def execute_tool(
        self,
        mcp_name: str,
        arguments: dict[str, Any],
        use_cache: bool = False,
    ) -> ToolExecutionResult:
        """执行指定的 MCP 工具。

        根据工具名称查找定义，并通过执行器调用注册的处理函数。

        Args:
            mcp_name: MCP 格式的工具名称（如 "mcp__top__query_station"）。
            arguments: 传递给工具处理器的参数字典。
            use_cache: 是否使用缓存结果，仅在执行器启用缓存时有效。

        Returns:
            ToolExecutionResult 实例，包含执行状态、结果或错误信息。

        Example:
            >>> result = await registry.execute_tool(
            ...     "mcp__top__query_station",
            ...     {"station_id": "ST001"}
            ... )
            >>> if result.success:
            ...     print(result.result)
            ... else:
            ...     print(f"Error: {result.error}")
        """
        tool = self._tools.get(mcp_name)
        if tool is None:
            logger.warning("Tool not found: %s", mcp_name)
            return ToolExecutionResult(
                success=False,
                error=f"Tool not found: {mcp_name}",
            )

        logger.debug(
            "Executing tool: mcp_name=%s, server=%s, use_cache=%s",
            mcp_name,
            tool.server_name,
            use_cache,
        )

        result = await self._executor.execute(
            server_name=tool.server_name,
            tool_name=tool.name,
            arguments=arguments,
            use_cache=use_cache,
        )

        if result.success:
            logger.debug(
                "Tool execution succeeded: mcp_name=%s, time_ms=%.2f",
                mcp_name,
                result.execution_time_ms,
            )
        else:
            logger.warning(
                "Tool execution failed: mcp_name=%s, error=%s",
                mcp_name,
                result.error,
            )

        return result

    # ==================== 便捷方法 ====================

    def register_local_tools(
        self,
        server_name: str,
        tools: list[dict[str, Any]],
    ) -> None:
        """批量注册本地工具。

        便捷方法，用于一次性注册多个工具。如果服务器不存在，
        会自动创建 local 类型的服务器配置。

        Args:
            server_name: 服务器名称。
            tools: 工具配置列表，每个工具字典应包含：
                - name (str): 工具名称（必需）
                - description (str): 工具描述（必需）
                - input_schema (dict): 输入参数 JSON Schema（必需）
                - handler (Callable): 异步处理函数（可选）
                - annotations (dict): 工具注解（可选）

        Example:
            >>> tools = [
            ...     {
            ...         "name": "query_station",
            ...         "description": "查询场站信息",
            ...         "input_schema": {"type": "object", ...},
            ...         "handler": query_handler
            ...     },
            ...     {
            ...         "name": "query_device",
            ...         "description": "查询设备信息",
            ...         "input_schema": {"type": "object", ...},
            ...         "handler": device_handler
            ...     }
            ... ]
            >>> registry.register_local_tools("top", tools)
        """
        # 确保服务器存在
        if server_name not in self._servers:
            self.register_server(
                MCPServerConfig(
                    name=server_name,
                    type="local",
                )
            )

        for tool_info in tools:
            self.register_tool(
                server_name=server_name,
                tool_name=tool_info["name"],
                description=tool_info["description"],
                input_schema=tool_info["input_schema"],
                handler=tool_info.get("handler"),
                annotations=tool_info.get("annotations"),
            )

        logger.info(
            "Batch registered %d tools for server: %s", len(tools), server_name
        )

    def get_stats(self) -> dict[str, Any]:
        """获取注册表统计信息。

        Returns:
            包含统计信息的字典，键包括：
                - total_servers: 服务器总数
                - enabled_servers: 启用的服务器数
                - servers_by_type: 按类型分组的服务器计数
                - total_tools: 工具总数
                - tools_by_server: 按服务器分组的工具计数
                - executor_stats: 执行器的统计信息

        Example:
            >>> stats = registry.get_stats()
            >>> print(f"服务器: {stats['total_servers']}, 工具: {stats['total_tools']}")
        """
        # 按类型统计服务器
        servers_by_type: dict[str, int] = {}
        for server in self._servers.values():
            servers_by_type[server.type] = servers_by_type.get(server.type, 0) + 1

        # 按服务器统计工具
        tools_by_server: dict[str, int] = {}
        for tool in self._tools.values():
            tools_by_server[tool.server_name] = (
                tools_by_server.get(tool.server_name, 0) + 1
            )

        return {
            "total_servers": len(self._servers),
            "enabled_servers": len(self.get_enabled_servers()),
            "servers_by_type": servers_by_type,
            "total_tools": len(self._tools),
            "tools_by_server": tools_by_server,
            "executor_stats": self._executor.get_stats(),
        }

    # ==================== 工具发现（预留） ====================

    async def discover_tools(self, server_name: str) -> list[dict[str, Any]]:
        """从服务器发现工具（预留接口）。

        用于从 HTTP 或 stdio 类型的 MCP Server 动态发现可用工具。
        当前实现返回空列表，待后续版本实现。

        Args:
            server_name: 服务器名称。

        Returns:
            发现的工具信息列表，当前始终返回空列表。

        Note:
            此接口为预留功能，计划支持：
                - HTTP 服务器：通过 /tools 端点发现
                - stdio 服务器：通过 MCP 协议发现
        """
        server = self._servers.get(server_name)
        if server is None:
            logger.warning("Server not found for tool discovery: %s", server_name)
            return []

        # TODO: 实现从 HTTP/stdio 服务器发现工具
        if server.type == "http":
            logger.debug("HTTP tool discovery not implemented for server: %s", server_name)
        elif server.type == "stdio":
            logger.debug("stdio tool discovery not implemented for server: %s", server_name)

        return []
