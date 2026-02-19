# -*- coding: utf-8 -*-
"""
MCP Tool Registry - MCP 工具注册表

统一管理所有 MCP 工具，支持：
- 工具注册与发现
- 服务器管理
- 与现有 SkillsRegistry 兼容
"""

from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
import logging

from .adapter import MCPToolAdapter, MCPToolDefinition
from .executor import MCPToolExecutor, ToolExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """
    MCP Server 配置

    Attributes:
        name: 服务器名称
        type: 服务器类型（"local" | "http" | "stdio"）
        description: 服务器描述
        enabled: 是否启用
        url: HTTP 服务器 URL（type=http 时）
        command: stdio 命令（type=stdio 时）
        env: 环境变量
        metadata: 额外元数据
    """
    name: str
    type: str = "local"
    description: str = ""
    enabled: bool = True
    url: Optional[str] = None
    command: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
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
    """
    MCP 工具注册表

    统一管理所有 MCP 工具，提供：
    - 工具注册与发现
    - 服务器生命周期管理
    - 工具执行
    - 与现有 SkillsRegistry 兼容的接口

    Usage:
        registry = MCPToolRegistry()

        # 注册服务器
        registry.register_server(MCPServerConfig(name="top", type="local"))

        # 注册工具
        registry.register_tool("top", "query_station", {
            "description": "查询场站信息",
            "input_schema": {...},
            "handler": async_func,
        })

        # 获取 Claude API 格式的工具列表
        tools = registry.get_claude_tools()

        # 执行工具
        result = await registry.execute_tool("mcp__top__query_station", {...})
    """

    def __init__(self, executor: Optional[MCPToolExecutor] = None):
        """
        初始化注册表

        Args:
            executor: 工具执行器（可选，默认创建新实例）
        """
        self._tools: Dict[str, MCPToolDefinition] = {}
        self._servers: Dict[str, MCPServerConfig] = {}
        self._executor = executor or MCPToolExecutor()
        self._initialized = False

    # ==================== 服务器管理 ====================

    def register_server(self, config: MCPServerConfig) -> None:
        """
        注册 MCP Server

        Args:
            config: 服务器配置
        """
        self._servers[config.name] = config
        logger.info(f"Registered MCP server: {config.name} (type={config.type})")

    def unregister_server(self, name: str) -> bool:
        """
        注销 MCP Server

        Args:
            name: 服务器名称

        Returns:
            是否成功注销
        """
        if name not in self._servers:
            return False

        del self._servers[name]

        # 移除该服务器的所有工具
        tools_to_remove = [
            mcp_name for mcp_name, tool in self._tools.items()
            if tool.server_name == name
        ]
        for mcp_name in tools_to_remove:
            del self._tools[mcp_name]

        logger.info(f"Unregistered MCP server: {name}")
        return True

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """获取服务器配置"""
        return self._servers.get(name)

    def get_all_servers(self) -> List[MCPServerConfig]:
        """获取所有服务器配置"""
        return list(self._servers.values())

    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """获取所有启用的服务器"""
        return [s for s in self._servers.values() if s.enabled]

    # ==================== 工具管理 ====================

    def register_tool(
        self,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Optional[Callable[..., Awaitable[Any]]] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> MCPToolDefinition:
        """
        注册 MCP 工具

        Args:
            server_name: MCP Server 名称
            tool_name: 工具名称
            description: 工具描述
            input_schema: 输入参数 JSON Schema
            handler: 处理函数（本地执行时需要）
            annotations: 工具注解

        Returns:
            MCPToolDefinition 实例
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
        if handler:
            self._executor.register_local_handler(
                server_name, tool_name, handler
            )

        logger.info(f"Registered MCP tool: {tool_def.mcp_name}")
        return tool_def

    def unregister_tool(self, mcp_name: str) -> bool:
        """
        注销工具

        Args:
            mcp_name: MCP 格式的工具名称

        Returns:
            是否成功注销
        """
        if mcp_name not in self._tools:
            return False

        tool = self._tools[mcp_name]
        del self._tools[mcp_name]

        # 移除处理器
        self._executor.unregister_local_handler(tool.server_name, tool.name)

        return True

    def get_tool(self, mcp_name: str) -> Optional[MCPToolDefinition]:
        """获取工具定义"""
        return self._tools.get(mcp_name)

    def get_all_tools(self) -> List[MCPToolDefinition]:
        """获取所有工具"""
        return list(self._tools.values())

    def get_tools_by_server(self, server_name: str) -> List[MCPToolDefinition]:
        """获取指定服务器的所有工具"""
        return [
            tool for tool in self._tools.values()
            if tool.server_name == server_name
        ]

    # ==================== Claude API 兼容 ====================

    def get_claude_tools(self) -> List[Dict[str, Any]]:
        """
        获取 Claude API 格式的工具列表

        Returns:
            Claude API 兼容的工具定义列表
        """
        return [
            MCPToolAdapter.to_claude_tool(tool)
            for tool in self._tools.values()
        ]

    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return list(self._tools.keys())

    # ==================== 工具执行 ====================

    async def execute_tool(
        self,
        mcp_name: str,
        arguments: Dict[str, Any],
        use_cache: bool = False,
    ) -> ToolExecutionResult:
        """
        执行工具

        Args:
            mcp_name: MCP 格式的工具名称
            arguments: 工具参数
            use_cache: 是否使用缓存

        Returns:
            ToolExecutionResult 实例
        """
        tool = self._tools.get(mcp_name)
        if not tool:
            return ToolExecutionResult(
                success=False,
                error=f"Tool not found: {mcp_name}",
            )

        return await self._executor.execute(
            server_name=tool.server_name,
            tool_name=tool.name,
            arguments=arguments,
            use_cache=use_cache,
        )

    # ==================== 便捷方法 ====================

    def register_local_tools(
        self,
        server_name: str,
        tools: List[Dict[str, Any]],
    ) -> None:
        """
        批量注册本地工具

        Args:
            server_name: 服务器名称
            tools: 工具列表，每个工具包含：
                - name: 工具名称
                - description: 描述
                - input_schema: 参数 Schema
                - handler: 处理函数
        """
        # 确保服务器存在
        if server_name not in self._servers:
            self.register_server(MCPServerConfig(
                name=server_name,
                type="local",
            ))

        for tool_info in tools:
            self.register_tool(
                server_name=server_name,
                tool_name=tool_info["name"],
                description=tool_info["description"],
                input_schema=tool_info["input_schema"],
                handler=tool_info.get("handler"),
                annotations=tool_info.get("annotations"),
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取注册表统计"""
        servers_by_type: Dict[str, int] = {}
        for server in self._servers.values():
            servers_by_type[server.type] = servers_by_type.get(server.type, 0) + 1

        tools_by_server: Dict[str, int] = {}
        for tool in self._tools.values():
            tools_by_server[tool.server_name] = tools_by_server.get(tool.server_name, 0) + 1

        return {
            "total_servers": len(self._servers),
            "enabled_servers": len(self.get_enabled_servers()),
            "servers_by_type": servers_by_type,
            "total_tools": len(self._tools),
            "tools_by_server": tools_by_server,
            "executor_stats": self._executor.get_stats(),
        }

    # ==================== 工具发现（预留） ====================

    async def discover_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """
        从服务器发现工具（预留接口）

        Args:
            server_name: 服务器名称

        Returns:
            工具信息列表
        """
        server = self._servers.get(server_name)
        if not server:
            return []

        # TODO: 实现从 HTTP/stdio 服务器发现工具
        if server.type == "http":
            # HTTP 发现
            pass
        elif server.type == "stdio":
            # stdio 发现
            pass

        return []
