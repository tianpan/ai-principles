# -*- coding: utf-8 -*-
"""
Unit tests for MCP Server handlers

Tests cover:
- TOP Server tools (query_station, query_device, get_pipeline_status, etc.)
- Knowledge Server tools (search_faq, get_emergency_guide, search_knowledge, etc.)
- Server creation functions
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from app.mcp.registry import MCPToolRegistry, MCPServerConfig
from app.mcp.servers.top_server import (
    query_station,
    query_device,
    get_pipeline_status,
    get_realtime_metrics,
    generate_daily_report,
    create_top_server,
    TOOLS_DEFINITION as TOP_TOOLS,
)
from app.mcp.servers.knowledge_server import (
    search_faq,
    get_emergency_guide,
    search_knowledge,
    get_gas_prediction,
    create_knowledge_server,
    TOOLS_DEFINITION as KNOWLEDGE_TOOLS,
    FAQ_DATA,
    EMERGENCY_GUIDES,
    KNOWLEDGE_BASE,
)


# ==================== TOP Server Tests ====================

class TestQueryStation:
    """Tests for query_station handler"""

    @pytest.mark.asyncio
    async def test_query_all_stations(self):
        """Test querying all stations"""
        result = await query_station()

        assert result["success"] is True
        assert "data" in result
        assert "count" in result
        assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_query_station_by_id(self):
        """Test querying station by ID"""
        result = await query_station(station_id="ST001")

        assert result["success"] is True
        assert result["data"]["id"] == "ST001"
        assert "name" in result["data"]
        assert "status" in result["data"]

    @pytest.mark.asyncio
    async def test_query_station_by_id_not_found(self):
        """Test querying non-existent station ID"""
        result = await query_station(station_id="NONEXISTENT")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_query_station_by_name(self):
        """Test querying station by name (fuzzy match)"""
        result = await query_station(station_name="龙岗")

        assert result["success"] is True
        assert "data" in result
        assert "count" in result
        assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_query_station_by_name_no_match(self):
        """Test querying station with name that doesn't match"""
        result = await query_station(station_name="不存在的场站名称")

        assert result["success"] is True
        assert result["count"] == 0
        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_station_data_structure(self):
        """Test station data has expected structure"""
        result = await query_station(station_id="ST001")

        station = result["data"]
        assert "id" in station
        assert "name" in station
        assert "type" in station
        assert "status" in station
        assert "address" in station
        assert "pressure_in" in station
        assert "pressure_out" in station
        assert "flow_rate" in station
        assert "last_update" in station


class TestQueryDevice:
    """Tests for query_device handler"""

    @pytest.mark.asyncio
    async def test_query_device_success(self):
        """Test querying existing device"""
        result = await query_device(device_id="DEV001")

        assert result["success"] is True
        assert result["data"]["id"] == "DEV001"
        assert "name" in result["data"]
        assert "type" in result["data"]
        assert "status" in result["data"]

    @pytest.mark.asyncio
    async def test_query_device_not_found(self):
        """Test querying non-existent device"""
        result = await query_device(device_id="NONEXISTENT")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_device_data_structure(self):
        """Test device data has expected structure"""
        result = await query_device(device_id="DEV001")

        device = result["data"]
        assert "id" in device
        assert "name" in device
        assert "type" in device
        assert "station_id" in device
        assert "status" in device
        assert "manufacturer" in device
        assert "install_date" in device
        assert "last_maintenance" in device
        assert "next_maintenance" in device
        assert "parameters" in device

    @pytest.mark.asyncio
    async def test_device_parameters_structure(self):
        """Test device parameters structure varies by device type"""
        # DEV001 is a regulator
        result = await query_device(device_id="DEV001")
        params = result["data"]["parameters"]

        assert "inlet_pressure" in params
        assert "outlet_pressure" in params
        assert "flow_capacity" in params

        # DEV002 is a flow meter
        result = await query_device(device_id="DEV002")
        params = result["data"]["parameters"]

        assert "type" in params
        assert "range" in params
        assert "accuracy" in params


class TestGetPipelineStatus:
    """Tests for get_pipeline_status handler"""

    @pytest.mark.asyncio
    async def test_get_all_pipelines(self):
        """Test getting all pipelines"""
        result = await get_pipeline_status()

        assert result["success"] is True
        assert "data" in result
        assert "count" in result
        assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_get_pipeline_by_id(self):
        """Test getting pipeline by ID"""
        result = await get_pipeline_status(pipeline_id="PL001")

        assert result["success"] is True
        assert result["data"]["id"] == "PL001"
        assert "name" in result["data"]
        assert "status" in result["data"]

    @pytest.mark.asyncio
    async def test_get_pipeline_not_found(self):
        """Test getting non-existent pipeline"""
        result = await get_pipeline_status(pipeline_id="NONEXISTENT")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_pipeline_data_structure(self):
        """Test pipeline data has expected structure"""
        result = await get_pipeline_status(pipeline_id="PL001")

        pipeline = result["data"]
        assert "id" in pipeline
        assert "name" in pipeline
        assert "diameter" in pipeline
        assert "material" in pipeline
        assert "length_km" in pipeline
        assert "pressure" in pipeline
        assert "flow_rate" in pipeline
        assert "status" in pipeline
        assert "leak_detection" in pipeline


class TestGetRealtimeMetrics:
    """Tests for get_realtime_metrics handler"""

    @pytest.mark.asyncio
    async def test_get_realtime_metrics_success(self):
        """Test getting realtime metrics"""
        result = await get_realtime_metrics(station_id="ST001")

        assert result["success"] is True
        assert result["data"]["station_id"] == "ST001"

    @pytest.mark.asyncio
    async def test_realtime_metrics_structure(self):
        """Test realtime metrics data structure"""
        result = await get_realtime_metrics(station_id="ST001")

        data = result["data"]
        assert "station_id" in data
        assert "timestamp" in data
        assert "metrics" in data
        assert "alerts" in data
        assert "status" in data

        metrics = data["metrics"]
        assert "pressure_in" in metrics
        assert "pressure_out" in metrics
        assert "temperature" in metrics
        assert "flow_rate" in metrics
        assert "valve_position" in metrics

    @pytest.mark.asyncio
    async def test_realtime_metrics_values_in_range(self):
        """Test that metric values are in reasonable ranges"""
        result = await get_realtime_metrics(station_id="ST001")

        metrics = result["data"]["metrics"]

        # Pressure should be positive
        assert metrics["pressure_in"] > 0
        assert metrics["pressure_out"] > 0

        # Temperature should be reasonable
        assert 0 < metrics["temperature"] < 50

        # Flow rate should be positive
        assert metrics["flow_rate"] > 0

        # Valve position should be 0-100
        assert 0 <= metrics["valve_position"] <= 100

    @pytest.mark.asyncio
    async def test_realtime_metrics_has_timestamp(self):
        """Test that realtime metrics includes timestamp"""
        result = await get_realtime_metrics(station_id="ST001")

        timestamp = result["data"]["timestamp"]
        # Should be valid ISO format
        parsed = datetime.fromisoformat(timestamp)
        assert isinstance(parsed, datetime)


class TestGenerateDailyReport:
    """Tests for generate_daily_report handler"""

    @pytest.mark.asyncio
    async def test_generate_report_default_date(self):
        """Test generating report with default date"""
        result = await generate_daily_report()

        assert result["success"] is True
        assert "data" in result

    @pytest.mark.asyncio
    async def test_generate_report_specific_date(self):
        """Test generating report for specific date"""
        result = await generate_daily_report(date="2024-01-15")

        assert result["success"] is True
        assert result["data"]["date"] == "2024-01-15"

    @pytest.mark.asyncio
    async def test_report_structure(self):
        """Test report data structure"""
        result = await generate_daily_report()

        data = result["data"]
        assert "date" in data
        assert "generated_at" in data
        assert "summary" in data
        assert "details" in data

        summary = data["summary"]
        assert "total_stations" in summary
        assert "active_stations" in summary
        assert "maintenance_stations" in summary
        assert "total_flow_m3" in summary
        assert "avg_pressure" in summary
        assert "incidents" in summary

    @pytest.mark.asyncio
    async def test_report_details_structure(self):
        """Test report details structure"""
        result = await generate_daily_report()

        for detail in result["data"]["details"]:
            assert "station" in detail
            assert "flow_m3" in detail
            assert "status" in detail


class TestCreateTopServer:
    """Tests for create_top_server function"""

    def test_create_top_server(self):
        """Test creating TOP server"""
        registry = MCPToolRegistry()

        result = create_top_server(registry)

        assert result == registry
        assert registry.get_server("top") is not None
        assert registry.get_server("top").type == "local"

    def test_create_top_server_registers_tools(self):
        """Test that create_top_server registers all tools"""
        registry = MCPToolRegistry()
        create_top_server(registry)

        tools = registry.get_tools_by_server("top")

        assert len(tools) == len(TOP_TOOLS)
        tool_names = {t.name for t in tools}
        expected_names = {"query_station", "query_device", "get_pipeline_status",
                         "get_realtime_metrics", "generate_daily_report"}
        assert tool_names == expected_names

    def test_top_tools_definition(self):
        """Test TOP tools have correct structure"""
        for tool in TOP_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert "handler" in tool
            assert callable(tool["handler"])


# ==================== Knowledge Server Tests ====================

class TestSearchFaq:
    """Tests for search_faq handler"""

    @pytest.mark.asyncio
    async def test_search_faq_all(self):
        """Test searching all FAQs"""
        result = await search_faq()

        assert result["success"] is True
        assert "data" in result
        assert "count" in result
        assert result["count"] == len(FAQ_DATA)

    @pytest.mark.asyncio
    async def test_search_faq_by_query(self):
        """Test searching FAQs by keyword"""
        result = await search_faq(query="泄漏")

        assert result["success"] is True
        assert result["count"] >= 1
        for faq in result["data"]:
            assert "泄漏" in faq["question"] or "泄漏" in faq["answer"]

    @pytest.mark.asyncio
    async def test_search_faq_by_category(self):
        """Test searching FAQs by category"""
        result = await search_faq(category="用气安全")

        assert result["success"] is True
        assert all(f["category"] == "用气安全" for f in result["data"])

    @pytest.mark.asyncio
    async def test_search_faq_combined_filters(self):
        """Test searching FAQs with both query and category"""
        result = await search_faq(query="缴费", category="缴费服务")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_faq_no_results(self):
        """Test searching with no matching results"""
        result = await search_faq(query="不存在的关键词xyz123")

        assert result["success"] is True
        assert result["count"] == 0
        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_faq_data_structure(self):
        """Test FAQ data structure"""
        result = await search_faq()

        for faq in result["data"]:
            assert "id" in faq
            assert "category" in faq
            assert "question" in faq
            assert "answer" in faq
            assert "keywords" in faq


class TestGetEmergencyGuide:
    """Tests for get_emergency_guide handler"""

    @pytest.mark.asyncio
    async def test_get_emergency_guide_pipeline_leak(self):
        """Test getting pipeline leak guide"""
        result = await get_emergency_guide(emergency_type="管道泄漏")

        assert result["success"] is True
        assert result["data"]["id"] == "EM001"
        assert result["data"]["type"] == "管道泄漏"
        assert "steps" in result["data"]
        assert "contact" in result["data"]

    @pytest.mark.asyncio
    async def test_get_emergency_guide_device_failure(self):
        """Test getting device failure guide"""
        result = await get_emergency_guide(emergency_type="设备故障")

        assert result["success"] is True
        assert result["data"]["type"] == "设备故障"

    @pytest.mark.asyncio
    async def test_get_emergency_guide_pressure_anomaly(self):
        """Test getting pressure anomaly guide"""
        result = await get_emergency_guide(emergency_type="压力异常")

        assert result["success"] is True
        assert result["data"]["type"] == "压力异常"

    @pytest.mark.asyncio
    async def test_get_emergency_guide_not_found(self):
        """Test getting guide for unknown emergency type"""
        result = await get_emergency_guide(emergency_type="未知类型")

        assert result["success"] is True
        assert "available_types" in result["data"]
        assert "all_guides" in result["data"]

    @pytest.mark.asyncio
    async def test_emergency_guide_structure(self):
        """Test emergency guide data structure"""
        result = await get_emergency_guide(emergency_type="管道泄漏")

        guide = result["data"]
        assert "id" in guide
        assert "type" in guide
        assert "severity" in guide
        assert "steps" in guide
        assert "contact" in guide
        assert isinstance(guide["steps"], list)
        assert len(guide["steps"]) > 0

    @pytest.mark.asyncio
    async def test_emergency_guide_has_timestamp(self):
        """Test emergency guide includes timestamp"""
        result = await get_emergency_guide(emergency_type="管道泄漏")

        assert "timestamp" in result
        timestamp = result["timestamp"]
        parsed = datetime.fromisoformat(timestamp)
        assert isinstance(parsed, datetime)


class TestSearchKnowledge:
    """Tests for search_knowledge handler"""

    @pytest.mark.asyncio
    async def test_search_knowledge_all(self):
        """Test searching all knowledge"""
        result = await search_knowledge(query="")  # Empty query returns all

        assert result["success"] is True
        assert "data" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_search_knowledge_by_query(self):
        """Test searching knowledge by keyword"""
        result = await search_knowledge(query="调压")

        assert result["success"] is True
        for item in result["data"]:
            assert "调压" in item["title"] or "调压" in item["content"]

    @pytest.mark.asyncio
    async def test_search_knowledge_by_category(self):
        """Test searching knowledge by category"""
        result = await search_knowledge(query="", category="设备知识")

        assert result["success"] is True
        for item in result["data"]:
            assert item["category"] == "设备知识"

    @pytest.mark.asyncio
    async def test_search_knowledge_no_results(self):
        """Test searching with no matching results"""
        result = await search_knowledge(query="不存在的关键词xyz789")

        assert result["success"] is True
        assert result["count"] == 0
        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_knowledge_data_structure(self):
        """Test knowledge base data structure"""
        for item in KNOWLEDGE_BASE:
            assert "id" in item
            assert "title" in item
            assert "content" in item
            assert "category" in item


class TestGetGasPrediction:
    """Tests for get_gas_prediction handler"""

    @pytest.mark.asyncio
    async def test_get_gas_prediction_default(self):
        """Test getting gas prediction with defaults"""
        result = await get_gas_prediction()

        assert result["success"] is True
        assert "data" in result

    @pytest.mark.asyncio
    async def test_get_gas_prediction_specific_date(self):
        """Test getting gas prediction for specific date"""
        result = await get_gas_prediction(date="2024-12-25")

        assert result["success"] is True
        assert result["data"]["date"] == "2024-12-25"

    @pytest.mark.asyncio
    async def test_get_gas_prediction_with_station(self):
        """Test getting gas prediction for specific station"""
        result = await get_gas_prediction(station_id="ST001")

        assert result["success"] is True
        assert result["data"]["station_id"] == "ST001"

    @pytest.mark.asyncio
    async def test_prediction_structure(self):
        """Test prediction data structure"""
        result = await get_gas_prediction()

        data = result["data"]
        assert "date" in data
        assert "station_id" in data
        assert "predicted_flow_m3" in data
        assert "confidence" in data
        assert "factors" in data
        assert "hourly_breakdown" in data
        assert "generated_at" in data

    @pytest.mark.asyncio
    async def test_prediction_factors(self):
        """Test prediction factors structure"""
        result = await get_gas_prediction()

        factors = result["data"]["factors"]
        assert "weekday_factor" in factors
        assert "temperature_factor" in factors

    @pytest.mark.asyncio
    async def test_prediction_hourly_breakdown(self):
        """Test hourly breakdown structure"""
        result = await get_gas_prediction()

        hourly = result["data"]["hourly_breakdown"]
        assert len(hourly) == 24  # 24 hours

        for hour_data in hourly:
            assert "hour" in hour_data
            assert "flow" in hour_data
            assert 0 <= hour_data["hour"] <= 23

    @pytest.mark.asyncio
    async def test_prediction_weekend_factor(self):
        """Test that weekend has lower prediction"""
        # Weekday (Monday = 0)
        weekday_result = await get_gas_prediction(date="2024-01-15")  # Monday
        # Weekend (Saturday = 5)
        weekend_result = await get_gas_prediction(date="2024-01-20")  # Saturday

        weekday_factor = weekday_result["data"]["factors"]["weekday_factor"]
        weekend_factor = weekend_result["data"]["factors"]["weekday_factor"]

        assert weekday_factor == 1.0
        assert weekend_factor == 0.9

    @pytest.mark.asyncio
    async def test_prediction_flow_reasonable(self):
        """Test that predicted flow is in reasonable range"""
        result = await get_gas_prediction()

        flow = result["data"]["predicted_flow_m3"]
        # Base flow is 45000, with factors applied should be in reasonable range
        assert 30000 < flow < 60000


class TestCreateKnowledgeServer:
    """Tests for create_knowledge_server function"""

    def test_create_knowledge_server(self):
        """Test creating knowledge server"""
        registry = MCPToolRegistry()

        result = create_knowledge_server(registry)

        assert result == registry
        assert registry.get_server("knowledge") is not None
        assert registry.get_server("knowledge").type == "local"

    def test_create_knowledge_server_registers_tools(self):
        """Test that create_knowledge_server registers all tools"""
        registry = MCPToolRegistry()
        create_knowledge_server(registry)

        tools = registry.get_tools_by_server("knowledge")

        assert len(tools) == len(KNOWLEDGE_TOOLS)
        tool_names = {t.name for t in tools}
        expected_names = {"search_faq", "get_emergency_guide",
                         "search_knowledge", "get_gas_prediction"}
        assert tool_names == expected_names

    def test_knowledge_tools_definition(self):
        """Test knowledge tools have correct structure"""
        for tool in KNOWLEDGE_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert "handler" in tool
            assert callable(tool["handler"])


# ==================== Server Integration Tests ====================

class TestServerIntegration:
    """Integration tests for server creation"""

    def test_both_servers_can_coexist(self):
        """Test that both servers can be registered together"""
        registry = MCPToolRegistry()

        create_top_server(registry)
        create_knowledge_server(registry)

        servers = registry.get_all_servers()
        assert len(servers) == 2
        server_names = {s.name for s in servers}
        assert server_names == {"top", "knowledge"}

    def test_all_tools_registered(self):
        """Test that all tools are registered"""
        registry = MCPToolRegistry()

        create_top_server(registry)
        create_knowledge_server(registry)

        all_tools = registry.get_all_tools()
        assert len(all_tools) == len(TOP_TOOLS) + len(KNOWLEDGE_TOOLS)

    def test_claude_tools_format(self):
        """Test that all tools can be converted to Claude format"""
        registry = MCPToolRegistry()

        create_top_server(registry)
        create_knowledge_server(registry)

        claude_tools = registry.get_claude_tools()

        for tool in claude_tools:
            assert "name" in tool
            assert tool["name"].startswith("mcp__")
            assert "description" in tool
            assert "input_schema" in tool

    @pytest.mark.asyncio
    async def test_execute_top_tools_via_registry(self):
        """Test executing TOP tools via registry"""
        registry = MCPToolRegistry()
        create_top_server(registry)

        # Test query_station
        result = await registry.execute_tool("mcp__top__query_station", {"station_id": "ST001"})
        assert result.success is True
        assert result.result["data"]["id"] == "ST001"

        # Test query_device
        result = await registry.execute_tool("mcp__top__query_device", {"device_id": "DEV001"})
        assert result.success is True
        assert result.result["data"]["id"] == "DEV001"

    @pytest.mark.asyncio
    async def test_execute_knowledge_tools_via_registry(self):
        """Test executing Knowledge tools via registry"""
        registry = MCPToolRegistry()
        create_knowledge_server(registry)

        # Test search_faq
        result = await registry.execute_tool("mcp__knowledge__search_faq", {"query": "泄漏"})
        assert result.success is True

        # Test get_emergency_guide
        result = await registry.execute_tool(
            "mcp__knowledge__get_emergency_guide",
            {"emergency_type": "管道泄漏"}
        )
        assert result.success is True
