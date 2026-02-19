# -*- coding: utf-8 -*-
"""
Unit tests for MCP Tool Registry module

Tests cover:
- MCPServerConfig dataclass
- MCPToolRegistry class
- register_server()
- unregister_server()
- register_tool()
- unregister_tool()
- get_claude_tools()
- execute_tool()
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.mcp.registry import MCPToolRegistry, MCPServerConfig
from app.mcp.adapter import MCPToolDefinition
from app.mcp.executor import ToolExecutionResult


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass"""

    def test_config_creation_basic(self):
        """Test basic server config creation"""
        config = MCPServerConfig(name="test_server")

        assert config.name == "test_server"
        assert config.type == "local"
        assert config.description == ""
        assert config.enabled is True
        assert config.url is None
        assert config.command is None
        assert config.env == {}
        assert config.metadata == {}

    def test_config_creation_full(self):
        """Test server config with all fields"""
        config = MCPServerConfig(
            name="http_server",
            type="http",
            description="HTTP MCP Server",
            enabled=True,
            url="http://localhost:8080",
            env={"API_KEY": "secret"},
            metadata={"version": "1.0"},
        )

        assert config.name == "http_server"
        assert config.type == "http"
        assert config.description == "HTTP MCP Server"
        assert config.enabled is True
        assert config.url == "http://localhost:8080"
        assert config.env == {"API_KEY": "secret"}
        assert config.metadata == {"version": "1.0"}

    def test_config_to_dict(self):
        """Test to_dict() method"""
        config = MCPServerConfig(
            name="test_server",
            type="local",
            description="Test Server",
            enabled=True,
            metadata={"key": "value"},
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "test_server"
        assert result["type"] == "local"
        assert result["description"] == "Test Server"
        assert result["enabled"] is True
        assert result["metadata"] == {"key": "value"}

    def test_config_disabled(self):
        """Test disabled server config"""
        config = MCPServerConfig(name="disabled_server", enabled=False)

        assert config.enabled is False

    def test_config_stdio_type(self):
        """Test stdio server config"""
        config = MCPServerConfig(
            name="stdio_server",
            type="stdio",
            command="python -m mcp_server",
            env={"DEBUG": "1"},
        )

        assert config.type == "stdio"
        assert config.command == "python -m mcp_server"


class TestMCPToolRegistryInit:
    """Tests for MCPToolRegistry initialization"""

    def test_init_default(self):
        """Test initialization with default executor"""
        registry = MCPToolRegistry()

        assert registry._tools == {}
        assert registry._servers == {}
        assert registry._executor is not None
        assert registry._initialized is False

    def test_init_with_executor(self):
        """Test initialization with custom executor"""
        from app.mcp.executor import MCPToolExecutor
        executor = MCPToolExecutor(timeout_seconds=60.0)

        registry = MCPToolRegistry(executor=executor)

        assert registry._executor == executor
        assert registry._executor.timeout == 60.0


class TestServerManagement:
    """Tests for server management methods"""

    def test_register_server(self, registry: MCPToolRegistry):
        """Test registering a server"""
        config = MCPServerConfig(name="test_server", type="local")

        registry.register_server(config)

        assert "test_server" in registry._servers
        assert registry._servers["test_server"] == config

    def test_register_multiple_servers(self, registry: MCPToolRegistry):
        """Test registering multiple servers"""
        registry.register_server(MCPServerConfig(name="server1"))
        registry.register_server(MCPServerConfig(name="server2"))

        assert len(registry._servers) == 2
        assert "server1" in registry._servers
        assert "server2" in registry._servers

    def test_register_server_overwrites(self, registry: MCPToolRegistry):
        """Test that registering same server overwrites"""
        config1 = MCPServerConfig(name="server", description="Version 1")
        config2 = MCPServerConfig(name="server", description="Version 2")

        registry.register_server(config1)
        registry.register_server(config2)

        assert registry._servers["server"].description == "Version 2"

    def test_unregister_server(self, registry: MCPToolRegistry):
        """Test unregistering a server"""
        registry.register_server(MCPServerConfig(name="server"))

        result = registry.unregister_server("server")

        assert result is True
        assert "server" not in registry._servers

    def test_unregister_nonexistent_server(self, registry: MCPToolRegistry):
        """Test unregistering non-existent server"""
        result = registry.unregister_server("nonexistent")

        assert result is False

    def test_unregister_server_removes_tools(self, registry: MCPToolRegistry, async_handler):
        """Test that unregistering server removes its tools"""
        registry.register_server(MCPServerConfig(name="server"))
        registry.register_tool(
            server_name="server",
            tool_name="tool",
            description="Test tool",
            input_schema={},
            handler=async_handler,
        )

        assert len(registry._tools) == 1

        registry.unregister_server("server")

        assert len(registry._tools) == 0

    def test_get_server(self, registry: MCPToolRegistry):
        """Test getting a server config"""
        config = MCPServerConfig(name="server", description="Test")
        registry.register_server(config)

        result = registry.get_server("server")

        assert result == config
        assert result.description == "Test"

    def test_get_server_nonexistent(self, registry: MCPToolRegistry):
        """Test getting non-existent server"""
        result = registry.get_server("nonexistent")

        assert result is None

    def test_get_all_servers(self, registry: MCPToolRegistry):
        """Test getting all servers"""
        registry.register_server(MCPServerConfig(name="server1"))
        registry.register_server(MCPServerConfig(name="server2"))

        servers = registry.get_all_servers()

        assert len(servers) == 2
        assert {s.name for s in servers} == {"server1", "server2"}

    def test_get_enabled_servers(self, registry: MCPToolRegistry):
        """Test getting enabled servers only"""
        registry.register_server(MCPServerConfig(name="enabled", enabled=True))
        registry.register_server(MCPServerConfig(name="disabled", enabled=False))

        servers = registry.get_enabled_servers()

        assert len(servers) == 1
        assert servers[0].name == "enabled"


class TestToolManagement:
    """Tests for tool management methods"""

    @pytest.mark.asyncio
    async def test_register_tool(self, registry: MCPToolRegistry, async_handler):
        """Test registering a tool"""
        registry.register_server(MCPServerConfig(name="server"))

        tool = registry.register_tool(
            server_name="server",
            tool_name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            handler=async_handler,
        )

        assert isinstance(tool, MCPToolDefinition)
        assert tool.name == "test_tool"
        assert tool.mcp_name == "mcp__server__test_tool"
        assert "mcp__server__test_tool" in registry._tools

    @pytest.mark.asyncio
    async def test_register_tool_without_handler(self, registry: MCPToolRegistry):
        """Test registering a tool without handler"""
        registry.register_server(MCPServerConfig(name="server"))

        tool = registry.register_tool(
            server_name="server",
            tool_name="no_handler_tool",
            description="Tool without handler",
            input_schema={"type": "object"},
        )

        assert tool is not None
        assert tool.name == "no_handler_tool"

    @pytest.mark.asyncio
    async def test_register_tool_with_annotations(self, registry: MCPToolRegistry):
        """Test registering a tool with annotations"""
        registry.register_server(MCPServerConfig(name="server"))

        tool = registry.register_tool(
            server_name="server",
            tool_name="annotated_tool",
            description="Annotated tool",
            input_schema={"type": "object"},
            annotations={"category": "query"},
        )

        assert tool.annotations == {"category": "query"}

    @pytest.mark.asyncio
    async def test_register_multiple_tools(self, registry: MCPToolRegistry, async_handler):
        """Test registering multiple tools"""
        registry.register_server(MCPServerConfig(name="server"))

        registry.register_tool("server", "tool1", "Tool 1", {}, async_handler)
        registry.register_tool("server", "tool2", "Tool 2", {}, async_handler)

        assert len(registry._tools) == 2

    @pytest.mark.asyncio
    async def test_unregister_tool(self, registry: MCPToolRegistry, async_handler):
        """Test unregistering a tool"""
        registry.register_server(MCPServerConfig(name="server"))
        registry.register_tool("server", "tool", "Test", {}, async_handler)

        result = registry.unregister_tool("mcp__server__tool")

        assert result is True
        assert "mcp__server__tool" not in registry._tools

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_tool(self, registry: MCPToolRegistry):
        """Test unregistering non-existent tool"""
        result = registry.unregister_tool("mcp__nonexistent__tool")

        assert result is False

    @pytest.mark.asyncio
    async def test_unregister_tool_removes_handler(
        self, registry: MCPToolRegistry, async_handler
    ):
        """Test that unregistering tool removes handler from executor"""
        registry.register_server(MCPServerConfig(name="server"))
        registry.register_tool("server", "tool", "Test", {}, async_handler)

        # Handler should be registered
        assert "server" in registry._executor._local_handlers

        registry.unregister_tool("mcp__server__tool")

        # Handler should be removed
        assert "tool" not in registry._executor._local_handlers.get("server", {})

    def test_get_tool(self, registry: MCPToolRegistry, async_handler):
        """Test getting a tool definition"""
        registry.register_server(MCPServerConfig(name="server"))
        tool = registry.register_tool("server", "tool", "Test", {}, async_handler)

        result = registry.get_tool("mcp__server__tool")

        assert result == tool

    def test_get_tool_nonexistent(self, registry: MCPToolRegistry):
        """Test getting non-existent tool"""
        result = registry.get_tool("mcp__nonexistent__tool")

        assert result is None

    def test_get_all_tools(self, registry: MCPToolRegistry, async_handler):
        """Test getting all tools"""
        registry.register_server(MCPServerConfig(name="server1"))
        registry.register_server(MCPServerConfig(name="server2"))
        registry.register_tool("server1", "tool1", "T1", {}, async_handler)
        registry.register_tool("server2", "tool2", "T2", {}, async_handler)

        tools = registry.get_all_tools()

        assert len(tools) == 2

    def test_get_tools_by_server(self, registry: MCPToolRegistry, async_handler):
        """Test getting tools by server"""
        registry.register_server(MCPServerConfig(name="server1"))
        registry.register_server(MCPServerConfig(name="server2"))
        registry.register_tool("server1", "tool1", "T1", {}, async_handler)
        registry.register_tool("server1", "tool2", "T2", {}, async_handler)
        registry.register_tool("server2", "tool3", "T3", {}, async_handler)

        tools = registry.get_tools_by_server("server1")

        assert len(tools) == 2
        assert all(t.server_name == "server1" for t in tools)


class TestClaudeTools:
    """Tests for Claude API compatibility methods"""

    @pytest.mark.asyncio
    async def test_get_claude_tools(self, registry: MCPToolRegistry, async_handler):
        """Test getting Claude format tools"""
        registry.register_server(MCPServerConfig(name="server"))
        registry.register_tool(
            "server",
            "test_tool",
            "A test tool",
            {"type": "object", "properties": {"param": {"type": "string"}}},
            async_handler,
        )

        tools = registry.get_claude_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "mcp__server__test_tool"
        assert tools[0]["description"] == "A test tool"
        assert "input_schema" in tools[0]

    @pytest.mark.asyncio
    async def test_get_claude_tools_empty(self, registry: MCPToolRegistry):
        """Test getting Claude tools when none registered"""
        tools = registry.get_claude_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_get_claude_tools_multiple(self, registry: MCPToolRegistry, async_handler):
        """Test getting multiple Claude format tools"""
        registry.register_server(MCPServerConfig(name="server"))
        registry.register_tool("server", "tool1", "Tool 1", {}, async_handler)
        registry.register_tool("server", "tool2", "Tool 2", {}, async_handler)

        tools = registry.get_claude_tools()

        assert len(tools) == 2
        assert {t["name"] for t in tools} == {"mcp__server__tool1", "mcp__server__tool2"}

    def test_get_tool_names(self, registry: MCPToolRegistry, async_handler):
        """Test getting tool names"""
        registry.register_server(MCPServerConfig(name="server"))
        registry.register_tool("server", "tool1", "T1", {}, async_handler)
        registry.register_tool("server", "tool2", "T2", {}, async_handler)

        names = registry.get_tool_names()

        assert len(names) == 2
        assert "mcp__server__tool1" in names
        assert "mcp__server__tool2" in names


class TestExecuteTool:
    """Tests for tool execution"""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, registry: MCPToolRegistry, async_handler):
        """Test successful tool execution"""
        registry.register_server(MCPServerConfig(name="server"))
        registry.register_tool("server", "tool", "Test", {}, async_handler)

        result = await registry.execute_tool(
            "mcp__server__tool",
            {"param": "value"},
        )

        assert result.success is True
        assert result.result == {"success": True, "data": {"param": "value"}}

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, registry: MCPToolRegistry):
        """Test executing non-existent tool"""
        result = await registry.execute_tool("mcp__nonexistent__tool", {})

        assert result.success is False
        assert "Tool not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_with_cache(self, registry: MCPToolRegistry, async_handler):
        """Test executing tool with cache enabled"""
        registry._executor.enable_cache = True
        registry.register_server(MCPServerConfig(name="server"))
        registry.register_tool("server", "tool", "Test", {}, async_handler)

        # First call
        result1 = await registry.execute_tool(
            "mcp__server__tool",
            {"param": "value"},
            use_cache=True,
        )

        # Second call with same args
        result2 = await registry.execute_tool(
            "mcp__server__tool",
            {"param": "value"},
            use_cache=True,
        )

        assert result1.success is True
        assert result2.success is True
        assert result2.source == "cache"

    @pytest.mark.asyncio
    async def test_execute_tool_fails(self, registry: MCPToolRegistry):
        """Test tool execution failure"""
        async def failing_handler(**kwargs):
            raise ValueError("Handler error")

        registry.register_server(MCPServerConfig(name="server"))
        registry.register_tool("server", "tool", "Test", {}, failing_handler)

        result = await registry.execute_tool("mcp__server__tool", {})

        assert result.success is False
        assert "Handler error" in result.error


class TestRegisterLocalTools:
    """Tests for register_local_tools() method"""

    @pytest.mark.asyncio
    async def test_register_local_tools(self, registry: MCPToolRegistry, async_handler):
        """Test batch registering local tools"""
        tools = [
            {
                "name": "tool1",
                "description": "Tool 1",
                "input_schema": {"type": "object"},
                "handler": async_handler,
            },
            {
                "name": "tool2",
                "description": "Tool 2",
                "input_schema": {"type": "object"},
                "handler": async_handler,
            },
        ]

        registry.register_local_tools("server", tools)

        assert "server" in registry._servers
        assert len(registry._tools) == 2
        assert "mcp__server__tool1" in registry._tools
        assert "mcp__server__tool2" in registry._tools

    @pytest.mark.asyncio
    async def test_register_local_tools_creates_server(self, registry: MCPToolRegistry):
        """Test that register_local_tools creates server if needed"""
        tools = [{"name": "tool", "description": "Test", "input_schema": {}}]

        registry.register_local_tools("new_server", tools)

        assert "new_server" in registry._servers
        assert registry._servers["new_server"].type == "local"

    @pytest.mark.asyncio
    async def test_register_local_tools_with_annotations(
        self, registry: MCPToolRegistry, async_handler
    ):
        """Test batch registering tools with annotations"""
        tools = [
            {
                "name": "tool",
                "description": "Tool",
                "input_schema": {},
                "handler": async_handler,
                "annotations": {"category": "test"},
            },
        ]

        registry.register_local_tools("server", tools)

        tool = registry.get_tool("mcp__server__tool")
        assert tool.annotations == {"category": "test"}


class TestGetStats:
    """Tests for get_stats() method"""

    @pytest.mark.asyncio
    async def test_get_stats_initial(self, registry: MCPToolRegistry):
        """Test initial statistics"""
        stats = registry.get_stats()

        assert stats["total_servers"] == 0
        assert stats["enabled_servers"] == 0
        assert stats["total_tools"] == 0
        assert stats["servers_by_type"] == {}
        assert stats["tools_by_server"] == {}

    @pytest.mark.asyncio
    async def test_get_stats_with_servers(self, registry: MCPToolRegistry):
        """Test statistics with servers"""
        registry.register_server(MCPServerConfig(name="local1", type="local"))
        registry.register_server(MCPServerConfig(name="local2", type="local"))
        registry.register_server(MCPServerConfig(name="http1", type="http"))

        stats = registry.get_stats()

        assert stats["total_servers"] == 3
        assert stats["servers_by_type"]["local"] == 2
        assert stats["servers_by_type"]["http"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_with_tools(self, registry: MCPToolRegistry, async_handler):
        """Test statistics with tools"""
        registry.register_server(MCPServerConfig(name="server1"))
        registry.register_server(MCPServerConfig(name="server2"))
        registry.register_tool("server1", "tool1", "T1", {}, async_handler)
        registry.register_tool("server1", "tool2", "T2", {}, async_handler)
        registry.register_tool("server2", "tool3", "T3", {}, async_handler)

        stats = registry.get_stats()

        assert stats["total_tools"] == 3
        assert stats["tools_by_server"]["server1"] == 2
        assert stats["tools_by_server"]["server2"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_enabled_servers(self, registry: MCPToolRegistry):
        """Test statistics for enabled servers count"""
        registry.register_server(MCPServerConfig(name="enabled", enabled=True))
        registry.register_server(MCPServerConfig(name="disabled", enabled=False))

        stats = registry.get_stats()

        assert stats["enabled_servers"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_includes_executor_stats(
        self, registry: MCPToolRegistry, async_handler
    ):
        """Test that stats include executor statistics"""
        registry.register_server(MCPServerConfig(name="server"))
        registry.register_tool("server", "tool", "Test", {}, async_handler)

        await registry.execute_tool("mcp__server__tool", {})

        stats = registry.get_stats()

        assert "executor_stats" in stats
        assert stats["executor_stats"]["total_calls"] == 1


class TestDiscoverTools:
    """Tests for discover_tools() method"""

    @pytest.mark.asyncio
    async def test_discover_tools_nonexistent_server(self, registry: MCPToolRegistry):
        """Test discovering tools from non-existent server"""
        result = await registry.discover_tools("nonexistent")

        assert result == []

    @pytest.mark.asyncio
    async def test_discover_tools_local_server(self, registry: MCPToolRegistry):
        """Test discovering tools from local server"""
        registry.register_server(MCPServerConfig(name="local", type="local"))

        result = await registry.discover_tools("local")

        # Local server discovery not implemented yet
        assert result == []

    @pytest.mark.asyncio
    async def test_discover_tools_http_server(self, registry: MCPToolRegistry):
        """Test discovering tools from HTTP server"""
        registry.register_server(MCPServerConfig(
            name="http",
            type="http",
            url="http://localhost:8080",
        ))

        result = await registry.discover_tools("http")

        # HTTP discovery not implemented yet
        assert result == []


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_workflow(self, registry: MCPToolRegistry, async_handler):
        """Test complete workflow: register server, tools, execute"""
        # Register server
        registry.register_server(MCPServerConfig(
            name="test_server",
            type="local",
            description="Test Server",
        ))

        # Register tools
        registry.register_tool(
            "test_server",
            "query",
            "Query tool",
            {"type": "object", "properties": {"id": {"type": "string"}}},
            async_handler,
        )

        # Get Claude tools
        claude_tools = registry.get_claude_tools()
        assert len(claude_tools) == 1

        # Execute tool
        result = await registry.execute_tool(
            "mcp__test_server__query",
            {"id": "123"},
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_multiple_servers_with_tools(
        self, registry: MCPToolRegistry, async_handler
    ):
        """Test multiple servers with different tools"""
        # Setup server1
        registry.register_server(MCPServerConfig(name="server1"))
        registry.register_tool("server1", "tool1", "T1", {}, async_handler)

        # Setup server2
        registry.register_server(MCPServerConfig(name="server2"))
        registry.register_tool("server2", "tool2", "T2", {}, async_handler)

        # Verify both servers and tools
        assert len(registry.get_all_servers()) == 2
        assert len(registry.get_all_tools()) == 2

        # Execute from each server
        result1 = await registry.execute_tool("mcp__server1__tool1", {})
        result2 = await registry.execute_tool("mcp__server2__tool2", {})

        assert result1.success is True
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_server_lifecycle(self, registry: MCPToolRegistry, async_handler):
        """Test server lifecycle: register, use, unregister"""
        # Register
        registry.register_server(MCPServerConfig(name="lifecycle"))
        registry.register_tool("lifecycle", "tool", "Test", {}, async_handler)

        # Use
        result = await registry.execute_tool("mcp__lifecycle__tool", {})
        assert result.success is True

        # Unregister
        registry.unregister_server("lifecycle")

        # Verify cleanup
        assert registry.get_server("lifecycle") is None
        assert registry.get_tool("mcp__lifecycle__tool") is None

        # Tool execution should fail
        result = await registry.execute_tool("mcp__lifecycle__tool", {})
        assert result.success is False
