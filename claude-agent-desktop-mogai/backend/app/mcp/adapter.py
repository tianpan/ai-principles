# -*- coding: utf-8 -*-
"""
MCP Tool Adapter - MCP 工具适配器

负责将 MCP 工具转换为 Claude API 兼容格式

Features:
    - 工具名称解析与生成
    - Claude API 格式转换
    - 工具定义创建
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MCPInvalidToolNameError(Exception):
    """MCP 无效工具名称异常

    当工具名称不符合 MCP 命名规范时抛出。
    MCP 工具名称格式: mcp__{server}__{tool}

    Attributes:
        tool_name: 无效的工具名称
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Invalid MCP tool name format: {tool_name}")


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
    input_schema: dict[str, Any]
    server_name: str
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典格式

        Returns:
            包含所有属性的字典
        """
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

    Class Attributes:
        MCP_PREFIX: MCP 工具名称前缀（"mcp__"）
    """

    MCP_PREFIX: str = "mcp__"

    @staticmethod
    def is_mcp_tool(tool_name: str) -> bool:
        """
        判断是否为 MCP 工具

        Args:
            tool_name: 工具名称

        Returns:
            如果是 MCP 工具返回 True，否则返回 False

        Example:
            >>> MCPToolAdapter.is_mcp_tool("mcp__top__query_station")
            True
            >>> MCPToolAdapter.is_mcp_tool("get_time")
            False
        """
        return tool_name.startswith(MCPToolAdapter.MCP_PREFIX)

    @staticmethod
    def to_claude_tool(mcp_tool: MCPToolDefinition) -> dict[str, Any]:
        """
        转换为 Claude API tool 格式

        Args:
            mcp_tool: MCP 工具定义

        Returns:
            Claude API 兼容的工具定义

        Example:
            >>> tool_def = MCPToolDefinition(
            ...     name="query",
            ...     mcp_name="mcp__top__query",
            ...     description="Query data",
            ...     input_schema={"type": "object"},
            ...     server_name="top",
            ... )
            >>> result = MCPToolAdapter.to_claude_tool(tool_def)
            >>> result["name"]
            'mcp__top__query'
        """
        tool_def: dict[str, Any] = {
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
            MCPInvalidToolNameError: 工具名称格式无效

        Example:
            >>> MCPToolAdapter.parse_tool_name("mcp__top__query_station")
            ('top', 'query_station')
        """
        if not MCPToolAdapter.is_mcp_tool(mcp_tool_name):
            raise MCPInvalidToolNameError(mcp_tool_name)

        # 移除前缀并分割
        name_without_prefix = mcp_tool_name[len(MCPToolAdapter.MCP_PREFIX) :]
        parts = name_without_prefix.split("__")

        if len(parts) < 2:
            raise MCPInvalidToolNameError(mcp_tool_name)

        server_name = parts[0]
        # 支持工具名中包含 __（如 "mcp__top__some__tool" -> "some__tool"）
        original_name = "__".join(parts[1:])

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

        Example:
            >>> MCPToolAdapter.make_mcp_name("top", "query_station")
            'mcp__top__query_station'
        """
        return f"{MCPToolAdapter.MCP_PREFIX}{server_name}__{tool_name}"

    @staticmethod
    def create_tool_definition(
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        annotations: dict[str, Any] | None = None,
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

        Example:
            >>> tool_def = MCPToolAdapter.create_tool_definition(
            ...     server_name="top",
            ...     tool_name="query_station",
            ...     description="Query station information",
            ...     input_schema={"type": "object", "properties": {}},
            ... )
            >>> tool_def.mcp_name
            'mcp__top__query_station'
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
