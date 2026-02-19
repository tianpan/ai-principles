# -*- coding: utf-8 -*-
"""
MCP Tool Adapter - MCP 工具适配器

负责将 MCP 工具转换为 Claude API 兼容格式
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MCPToolDefinition:
    """
    MCP 工具定义

    Attributes:
        name: 原始工具名称（如 "query_station"）
        mcp_name: MCP 格式名称（如 "mcp__top__query_station"）
        description: 工具描述
        input_schema: 输入参数 JSON Schema
        server_name: 来源 MCP Server 名称
        annotations: 工具注解
    """
    name: str
    mcp_name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    annotations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "mcp_name": self.mcp_name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server_name": self.server_name,
            "annotations": self.annotations,
        }


class MCPToolAdapter:
    """
    MCP 工具适配器

    负责将 MCP 工具转换为 Claude API 兼容格式
    """

    # MCP 工具名称前缀
    MCP_PREFIX = "mcp__"

    @staticmethod
    def is_mcp_tool(tool_name: str) -> bool:
        """
        判断是否为 MCP 工具

        Args:
            tool_name: 工具名称

        Returns:
            是否为 MCP 工具
        """
        return tool_name.startswith(MCPToolAdapter.MCP_PREFIX)

    @staticmethod
    def to_claude_tool(mcp_tool: MCPToolDefinition) -> Dict[str, Any]:
        """
        转换为 Claude API tool 格式

        Args:
            mcp_tool: MCP 工具定义

        Returns:
            Claude API 兼容的工具定义
        """
        tool_def = {
            "name": mcp_tool.mcp_name,
            "description": mcp_tool.description,
            "input_schema": mcp_tool.input_schema,
        }

        # 添加注解（如果存在）
        if mcp_tool.annotations:
            tool_def["annotations"] = mcp_tool.annotations

        return tool_def

    @staticmethod
    def parse_tool_name(mcp_tool_name: str) -> tuple[str, str]:
        """
        解析 MCP 工具名称

        将 "mcp__top__query_station" 解析为 ("top", "query_station")

        Args:
            mcp_tool_name: MCP 格式的工具名称

        Returns:
            (server_name, original_name) 元组

        Raises:
            ValueError: 工具名称格式无效
        """
        if not MCPToolAdapter.is_mcp_tool(mcp_tool_name):
            raise ValueError(f"Not an MCP tool: {mcp_tool_name}")

        # 移除前缀并分割
        parts = mcp_tool_name[len(MCPToolAdapter.MCP_PREFIX):].split("__")

        if len(parts) < 2:
            raise ValueError(f"Invalid MCP tool name format: {mcp_tool_name}")

        server_name = parts[0]
        original_name = "__".join(parts[1:])  # 支持工具名中包含 __

        return server_name, original_name

    @staticmethod
    def make_mcp_name(server_name: str, tool_name: str) -> str:
        """
        构建 MCP 工具名称

        Args:
            server_name: MCP Server 名称
            tool_name: 原始工具名称

        Returns:
            MCP 格式的工具名称
        """
        return f"{MCPToolAdapter.MCP_PREFIX}{server_name}__{tool_name}"

    @staticmethod
    def create_tool_definition(
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: Dict[str, Any],
        annotations: Optional[Dict[str, Any]] = None
    ) -> MCPToolDefinition:
        """
        创建 MCP 工具定义

        Args:
            server_name: MCP Server 名称
            tool_name: 原始工具名称
            description: 工具描述
            input_schema: 输入参数 Schema
            annotations: 工具注解（可选）

        Returns:
            MCPToolDefinition 实例
        """
        mcp_name = MCPToolAdapter.make_mcp_name(server_name, tool_name)

        return MCPToolDefinition(
            name=tool_name,
            mcp_name=mcp_name,
            description=description,
            input_schema=input_schema,
            server_name=server_name,
            annotations=annotations or {},
        )
