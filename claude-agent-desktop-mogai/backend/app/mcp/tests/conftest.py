# -*- coding: utf-8 -*-
"""
Pytest fixtures for MCP module tests
"""

import pytest
import asyncio
from typing import Dict, Any

from app.mcp.adapter import MCPToolAdapter, MCPToolDefinition
from app.mcp.executor import MCPToolExecutor, ToolExecutionResult
from app.mcp.registry import MCPToolRegistry, MCPServerConfig


# ==================== Async Support ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==================== Adapter Fixtures ====================

@pytest.fixture
def sample_tool_definition() -> MCPToolDefinition:
    """Sample MCP tool definition for testing"""
    return MCPToolDefinition(
        name="query_station",
        mcp_name="mcp__top__query_station",
        description="Query station information",
        input_schema={
            "type": "object",
            "properties": {
                "station_id": {"type": "string"},
            },
        },
        server_name="top",
        annotations={"category": "query"},
    )


@pytest.fixture
def sample_input_schema() -> Dict[str, Any]:
    """Sample input schema for testing"""
    return {
        "type": "object",
        "properties": {
            "station_id": {
                "type": "string",
                "description": "Station ID",
            },
            "station_name": {
                "type": "string",
                "description": "Station name for fuzzy matching",
            },
        },
        "required": [],
    }


# ==================== Executor Fixtures ====================

@pytest.fixture
def executor() -> MCPToolExecutor:
    """Create MCPToolExecutor instance"""
    return MCPToolExecutor(
        timeout_seconds=5.0,
        max_retries=2,
        enable_cache=True,
    )


@pytest.fixture
def executor_no_cache() -> MCPToolExecutor:
    """Create MCPToolExecutor instance without cache"""
    return MCPToolExecutor(
        timeout_seconds=5.0,
        max_retries=2,
        enable_cache=False,
    )


@pytest.fixture
async def async_handler():
    """Sample async handler for testing"""
    async def _handler(**kwargs) -> Dict[str, Any]:
        await asyncio.sleep(0.01)  # Simulate async work
        return {"success": True, "data": kwargs}
    return _handler


@pytest.fixture
async def slow_async_handler():
    """Slow async handler for timeout testing"""
    async def _handler(**kwargs) -> Dict[str, Any]:
        await asyncio.sleep(10)  # Simulate slow operation
        return {"success": True, "data": kwargs}
    return _handler


@pytest.fixture
async def failing_async_handler():
    """Failing async handler for error testing"""
    async def _handler(**kwargs) -> Dict[str, Any]:
        raise ValueError("Intentional test error")
    return _handler


# ==================== Registry Fixtures ====================

@pytest.fixture
def registry() -> MCPToolRegistry:
    """Create MCPToolRegistry instance"""
    return MCPToolRegistry()


@pytest.fixture
def registry_with_tools(registry: MCPToolRegistry, async_handler) -> MCPToolRegistry:
    """Create MCPToolRegistry with pre-registered tools"""
    # Register server
    registry.register_server(MCPServerConfig(
        name="test_server",
        type="local",
        description="Test server",
        enabled=True,
    ))

    # Register tools
    registry.register_tool(
        server_name="test_server",
        tool_name="test_tool",
        description="A test tool",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
            },
        },
        handler=async_handler,
    )

    return registry


@pytest.fixture
def sample_server_config() -> MCPServerConfig:
    """Sample server configuration for testing"""
    return MCPServerConfig(
        name="top",
        type="local",
        description="TOP Server - Station Management",
        enabled=True,
        url=None,
        command=None,
        env={},
        metadata={"version": "1.0"},
    )


@pytest.fixture
def http_server_config() -> MCPServerConfig:
    """HTTP server configuration for testing"""
    return MCPServerConfig(
        name="http_server",
        type="http",
        description="HTTP MCP Server",
        enabled=True,
        url="http://localhost:8080/mcp",
        env={"API_KEY": "test_key"},
    )


@pytest.fixture
def stdio_server_config() -> MCPServerConfig:
    """Stdio server configuration for testing"""
    return MCPServerConfig(
        name="stdio_server",
        type="stdio",
        description="Stdio MCP Server",
        enabled=True,
        command="python",
        env={"DEBUG": "1"},
    )


# ==================== Server Handler Fixtures ====================

@pytest.fixture
def top_handlers():
    """TOP server handler functions"""
    from app.mcp.servers.top_server import (
        query_station,
        query_device,
        get_pipeline_status,
        get_realtime_metrics,
        generate_daily_report,
    )
    return {
        "query_station": query_station,
        "query_device": query_device,
        "get_pipeline_status": get_pipeline_status,
        "get_realtime_metrics": get_realtime_metrics,
        "generate_daily_report": generate_daily_report,
    }


@pytest.fixture
def knowledge_handlers():
    """Knowledge server handler functions"""
    from app.mcp.servers.knowledge_server import (
        search_faq,
        get_emergency_guide,
        search_knowledge,
        get_gas_prediction,
    )
    return {
        "search_faq": search_faq,
        "get_emergency_guide": get_emergency_guide,
        "search_knowledge": search_knowledge,
        "get_gas_prediction": get_gas_prediction,
    }


# ==================== Helper Functions ====================

@pytest.fixture
def create_tool_definition_helper():
    """Helper to create tool definitions"""
    def _create(
        server_name: str = "test",
        tool_name: str = "test_tool",
        description: str = "Test tool",
        input_schema: Dict[str, Any] = None,
    ) -> MCPToolDefinition:
        return MCPToolAdapter.create_tool_definition(
            server_name=server_name,
            tool_name=tool_name,
            description=description,
            input_schema=input_schema or {"type": "object"},
        )
    return _create
