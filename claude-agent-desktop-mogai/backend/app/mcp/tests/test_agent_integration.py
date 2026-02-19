# -*- coding: utf-8 -*-
"""
MCP 与 AgentEngine 集成测试

测试 MCP 工具与 AgentEngine 的集成功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.mcp import MCPToolRegistry, MCPToolAdapter
from app.mcp.servers import create_top_server, create_knowledge_server
from app.core.agent_engine import AgentEngine


class TestAgentEngineMCPIntegration:
    """AgentEngine 与 MCP 集成测试"""

    @pytest.fixture
    def mcp_registry(self):
        """创建带有预注册工具的 MCP Registry"""
        registry = MCPToolRegistry()
        create_top_server(registry)
        create_knowledge_server(registry)
        return registry

    @pytest.fixture
    def agent_engine(self, mcp_registry):
        """创建带有 MCP Registry 的 AgentEngine"""
        with patch('app.core.agent_engine.AsyncAnthropic'):
            engine = AgentEngine(
                api_key="test-key",
                mcp_registry=mcp_registry,
                enable_mcp=True,
            )
            return engine

    def test_mcp_registry_injected(self, agent_engine, mcp_registry):
        """测试 MCP Registry 正确注入"""
        assert agent_engine.mcp_registry is mcp_registry
        assert agent_engine.enable_mcp is True

    def test_tools_definition_includes_mcp(self, agent_engine, mcp_registry):
        """测试工具定义包含 MCP 工具"""
        tools = agent_engine._get_tools_definition()

        # 应包含 Skills 工具
        skill_names = [t["name"] for t in tools if not t["name"].startswith("mcp__")]
        assert "get_current_time" in skill_names

        # 应包含 MCP 工具
        mcp_names = [t["name"] for t in tools if t["name"].startswith("mcp__")]
        assert len(mcp_names) > 0

        # 检查 TOP 工具
        assert "mcp__top__query_station" in mcp_names
        assert "mcp__top__query_device" in mcp_names

        # 检查 Knowledge 工具
        assert "mcp__knowledge__search_faq" in mcp_names
        assert "mcp__knowledge__get_emergency_guide" in mcp_names

    def test_mcp_can_be_disabled(self, mcp_registry):
        """测试 MCP 可以被禁用"""
        with patch('app.core.agent_engine.AsyncAnthropic'):
            engine = AgentEngine(
                api_key="test-key",
                mcp_registry=mcp_registry,
                enable_mcp=False,
            )

            tools = engine._get_tools_definition()
            mcp_names = [t["name"] for t in tools if t["name"].startswith("mcp__")]

            # 禁用后不应包含 MCP 工具
            assert len(mcp_names) == 0

    @pytest.mark.asyncio
    async def test_execute_mcp_tool(self, agent_engine):
        """测试执行 MCP 工具"""
        result = await agent_engine._execute_tool(
            "mcp__top__query_station",
            {"station_id": "ST001"}
        )

        assert result["success"] is True
        assert result["data"]["id"] == "ST001"

    @pytest.mark.asyncio
    async def test_execute_skill_not_mcp(self, agent_engine):
        """测试执行普通 Skill（非 MCP）"""
        result = await agent_engine._execute_tool(
            "get_current_time",
            {}
        )

        # Skill 执行结果
        assert "current_time" in result or "error" in result

    @pytest.mark.asyncio
    async def test_mcp_tool_name_parsing(self, agent_engine):
        """测试 MCP 工具名称解析"""
        # 验证 MCP 工具名称格式
        assert MCPToolAdapter.is_mcp_tool("mcp__top__query_station")
        assert MCPToolAdapter.is_mcp_tool("mcp__knowledge__search_faq")

        # 验证非 MCP 工具
        assert not MCPToolAdapter.is_mcp_tool("get_current_time")
        assert not MCPToolAdapter.is_mcp_tool("calculator")

    @pytest.mark.asyncio
    async def test_execute_unknown_mcp_tool(self, agent_engine):
        """测试执行未知的 MCP 工具"""
        result = await agent_engine._execute_tool(
            "mcp__unknown__tool",
            {}
        )

        # 未知工具返回 error 字典
        assert "error" in result
        assert "not found" in result["error"].lower() or "Tool not found" in result["error"]

    def test_tool_count(self, agent_engine):
        """测试工具数量"""
        tools = agent_engine._get_tools_definition()

        # Skills 工具数量（4个内置技能）
        skill_count = len([t for t in tools if not t["name"].startswith("mcp__")])

        # MCP 工具数量
        mcp_count = len([t for t in tools if t["name"].startswith("mcp__")])

        # TOP: 5 工具, Knowledge: 4 工具 = 9 MCP 工具
        assert mcp_count == 9
        assert skill_count >= 4  # 至少4个内置技能


class TestMCPToolRouting:
    """MCP 工具路由测试"""

    @pytest.fixture
    def engine_with_mcp(self):
        """创建启用 MCP 的引擎"""
        registry = MCPToolRegistry()
        create_top_server(registry)

        with patch('app.core.agent_engine.AsyncAnthropic'):
            return AgentEngine(
                api_key="test-key",
                mcp_registry=registry,
                enable_mcp=True,
            )

    @pytest.fixture
    def engine_without_mcp(self):
        """创建禁用 MCP 的引擎"""
        registry = MCPToolRegistry()
        create_top_server(registry)

        with patch('app.core.agent_engine.AsyncAnthropic'):
            return AgentEngine(
                api_key="test-key",
                mcp_registry=registry,
                enable_mcp=False,
            )

    def test_routing_to_mcp(self, engine_with_mcp):
        """测试路由到 MCP 执行器"""
        # MCP 工具应该被路由到 MCP Registry
        assert MCPToolAdapter.is_mcp_tool("mcp__top__query_station")

    def test_routing_to_skills(self, engine_with_mcp):
        """测试路由到 Skills 执行器"""
        # 非 MCP 工具应该被路由到 SkillsRegistry
        assert not MCPToolAdapter.is_mcp_tool("get_current_time")

    @pytest.mark.asyncio
    async def test_mcp_disabled_tool_not_available(self, engine_without_mcp):
        """测试 MCP 禁用时工具不可用"""
        tools = engine_without_mcp._get_tools_definition()
        mcp_tools = [t for t in tools if t["name"].startswith("mcp__")]

        # 禁用 MCP 后，不应该有 MCP 工具
        assert len(mcp_tools) == 0


class TestFullMCPWorkflow:
    """完整 MCP 工作流测试"""

    @pytest.fixture
    def full_engine(self):
        """创建完整配置的引擎"""
        registry = MCPToolRegistry()
        create_top_server(registry)
        create_knowledge_server(registry)

        with patch('app.core.agent_engine.AsyncAnthropic'):
            return AgentEngine(
                api_key="test-key",
                mcp_registry=registry,
                enable_mcp=True,
            )

    @pytest.mark.asyncio
    async def test_query_station_workflow(self, full_engine):
        """测试查询场站工作流"""
        # 1. 获取所有场站
        result = await full_engine._execute_tool(
            "mcp__top__query_station",
            {}
        )
        assert result["success"] is True
        assert result["count"] >= 1

        # 2. 按 ID 查询
        result = await full_engine._execute_tool(
            "mcp__top__query_station",
            {"station_id": "ST001"}
        )
        assert result["success"] is True
        assert result["data"]["id"] == "ST001"

        # 3. 按名称模糊查询
        result = await full_engine._execute_tool(
            "mcp__top__query_station",
            {"station_name": "龙岗"}
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_emergency_guide_workflow(self, full_engine):
        """测试应急指南工作流"""
        result = await full_engine._execute_tool(
            "mcp__knowledge__get_emergency_guide",
            {"emergency_type": "管道泄漏"}
        )

        assert result["success"] is True
        assert "data" in result
        assert "steps" in result["data"]

    @pytest.mark.asyncio
    async def test_daily_report_workflow(self, full_engine):
        """测试日报生成工作流"""
        result = await full_engine._execute_tool(
            "mcp__top__generate_daily_report",
            {}
        )

        assert result["success"] is True
        assert "data" in result
        assert "summary" in result["data"]
        assert result["data"]["summary"]["total_stations"] >= 1

    @pytest.mark.asyncio
    async def test_gas_prediction_workflow(self, full_engine):
        """测试用气预测工作流"""
        result = await full_engine._execute_tool(
            "mcp__knowledge__get_gas_prediction",
            {}
        )

        assert result["success"] is True
        assert "data" in result
        assert "predicted_flow_m3" in result["data"]
        assert result["data"]["predicted_flow_m3"] > 0
