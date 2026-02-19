# -*- coding: utf-8 -*-
"""
TOP MCP Server - 港华运营平台 MCP 服务器

提供场站查询、设备监控、管网管理等工具
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import random

from ..registry import MCPToolRegistry, MCPServerConfig


# ==================== 工具处理器 ====================

async def query_station(station_id: Optional[str] = None, station_name: Optional[str] = None) -> Dict[str, Any]:
    """
    查询场站信息

    Args:
        station_id: 场站 ID
        station_name: 场站名称（模糊匹配）

    Returns:
        场站信息
    """
    # Mock 数据 - 实际应调用 TOP API
    stations = [
        {
            "id": "ST001",
            "name": "深圳港华燃气总部站",
            "type": "调压站",
            "status": "正常运行",
            "address": "深圳市福田区福华路 100 号",
            "pressure_in": 0.4,
            "pressure_out": 0.02,
            "flow_rate": 12500,
            "last_update": datetime.now().isoformat(),
        },
        {
            "id": "ST002",
            "name": "龙岗中心站",
            "type": "门站",
            "status": "正常运行",
            "address": "深圳市龙岗区龙城大道 50 号",
            "pressure_in": 0.6,
            "pressure_out": 0.35,
            "flow_rate": 28000,
            "last_update": datetime.now().isoformat(),
        },
        {
            "id": "ST003",
            "name": "宝安西乡站",
            "type": "调压站",
            "status": "维护中",
            "address": "深圳市宝安区西乡大道 200 号",
            "pressure_in": 0.35,
            "pressure_out": 0.02,
            "flow_rate": 8000,
            "last_update": datetime.now().isoformat(),
        },
    ]

    if station_id:
        for station in stations:
            if station["id"] == station_id:
                return {"success": True, "data": station}
        return {"success": False, "error": f"未找到场站: {station_id}"}

    if station_name:
        results = [s for s in stations if station_name in s["name"]]
        return {"success": True, "data": results, "count": len(results)}

    return {"success": True, "data": stations, "count": len(stations)}


async def query_device(device_id: str) -> Dict[str, Any]:
    """
    查询设备信息

    Args:
        device_id: 设备 ID

    Returns:
        设备信息
    """
    # Mock 数据
    devices = {
        "DEV001": {
            "id": "DEV001",
            "name": "调压器 A",
            "type": "调压器",
            "station_id": "ST001",
            "status": "正常运行",
            "manufacturer": "费希尔",
            "install_date": "2020-06-15",
            "last_maintenance": "2024-01-20",
            "next_maintenance": "2024-07-20",
            "parameters": {
                "inlet_pressure": 0.4,
                "outlet_pressure": 0.02,
                "flow_capacity": 5000,
            },
        },
        "DEV002": {
            "id": "DEV002",
            "name": "流量计 B",
            "type": "流量计",
            "station_id": "ST001",
            "status": "正常运行",
            "manufacturer": "埃尔斯特",
            "install_date": "2021-03-10",
            "last_maintenance": "2024-02-15",
            "next_maintenance": "2024-08-15",
            "parameters": {
                "type": "涡轮流量计",
                "range": "0-10000 m³/h",
                "accuracy": "±1%",
            },
        },
    }

    if device_id in devices:
        return {"success": True, "data": devices[device_id]}
    return {"success": False, "error": f"未找到设备: {device_id}"}


async def get_pipeline_status(pipeline_id: Optional[str] = None) -> Dict[str, Any]:
    """
    获取管网状态

    Args:
        pipeline_id: 管网 ID（可选，不传则返回所有）

    Returns:
        管网状态信息
    """
    # Mock 数据
    pipelines = [
        {
            "id": "PL001",
            "name": "福华主管道",
            "diameter": "DN300",
            "material": "PE",
            "length_km": 5.2,
            "pressure": 0.35,
            "flow_rate": 15000,
            "status": "正常运行",
            "leak_detection": "无泄漏",
        },
        {
            "id": "PL002",
            "name": "龙岗支线",
            "diameter": "DN200",
            "material": "钢管",
            "length_km": 12.8,
            "pressure": 0.30,
            "flow_rate": 8000,
            "status": "正常运行",
            "leak_detection": "无泄漏",
        },
    ]

    if pipeline_id:
        for p in pipelines:
            if p["id"] == pipeline_id:
                return {"success": True, "data": p}
        return {"success": False, "error": f"未找到管网: {pipeline_id}"}

    return {"success": True, "data": pipelines, "count": len(pipelines)}


async def get_realtime_metrics(station_id: str) -> Dict[str, Any]:
    """
    获取实时监控指标

    Args:
        station_id: 场站 ID

    Returns:
        实时监控数据
    """
    # Mock 实时数据
    return {
        "success": True,
        "data": {
            "station_id": station_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "pressure_in": round(0.4 + random.uniform(-0.02, 0.02), 3),
                "pressure_out": round(0.02 + random.uniform(-0.002, 0.002), 4),
                "temperature": round(25 + random.uniform(-2, 2), 1),
                "flow_rate": int(12500 + random.randint(-500, 500)),
                "valve_position": random.randint(40, 60),
            },
            "alerts": [],
            "status": "正常运行",
        },
    }


async def generate_daily_report(date: Optional[str] = None) -> Dict[str, Any]:
    """
    生成运营日报

    Args:
        date: 日期（YYYY-MM-DD，默认今天）

    Returns:
        日报数据
    """
    report_date = date or datetime.now().strftime("%Y-%m-%d")

    return {
        "success": True,
        "data": {
            "date": report_date,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_stations": 3,
                "active_stations": 2,
                "maintenance_stations": 1,
                "total_flow_m3": 48500,
                "avg_pressure": 0.35,
                "incidents": 0,
            },
            "details": [
                {
                    "station": "深圳港华燃气总部站",
                    "flow_m3": 12500,
                    "status": "正常运行",
                },
                {
                    "station": "龙岗中心站",
                    "flow_m3": 28000,
                    "status": "正常运行",
                },
                {
                    "station": "宝安西乡站",
                    "flow_m3": 8000,
                    "status": "维护中",
                },
            ],
        },
    }


# ==================== 工具定义 ====================

TOOLS_DEFINITION = [
    {
        "name": "query_station",
        "description": "查询场站信息，支持按 ID 或名称查询",
        "input_schema": {
            "type": "object",
            "properties": {
                "station_id": {
                    "type": "string",
                    "description": "场站 ID，如 ST001",
                },
                "station_name": {
                    "type": "string",
                    "description": "场站名称（支持模糊匹配）",
                },
            },
        },
        "handler": query_station,
    },
    {
        "name": "query_device",
        "description": "查询设备信息，包括设备状态、维护记录等",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "设备 ID，如 DEV001",
                },
            },
            "required": ["device_id"],
        },
        "handler": query_device,
    },
    {
        "name": "get_pipeline_status",
        "description": "获取管网运行状态，包括压力、流量、泄漏检测等",
        "input_schema": {
            "type": "object",
            "properties": {
                "pipeline_id": {
                    "type": "string",
                    "description": "管网 ID（可选）",
                },
            },
        },
        "handler": get_pipeline_status,
    },
    {
        "name": "get_realtime_metrics",
        "description": "获取场站实时监控指标，包括压力、温度、流量等",
        "input_schema": {
            "type": "object",
            "properties": {
                "station_id": {
                    "type": "string",
                    "description": "场站 ID",
                },
            },
            "required": ["station_id"],
        },
        "handler": get_realtime_metrics,
    },
    {
        "name": "generate_daily_report",
        "description": "生成运营日报，包含各场站的流量、状态汇总",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "报告日期（YYYY-MM-DD），默认今天",
                },
            },
        },
        "handler": generate_daily_report,
    },
]


# ==================== 服务器创建 ====================

def create_top_server(registry: MCPToolRegistry) -> MCPToolRegistry:
    """
    创建 TOP MCP Server 并注册到 Registry

    Args:
        registry: MCP 工具注册表

    Returns:
        注册了 TOP 工具的 Registry
    """
    # 注册服务器
    registry.register_server(MCPServerConfig(
        name="top",
        type="local",
        description="港华运营平台 - 场站、设备、管网管理",
        enabled=True,
    ))

    # 注册工具
    for tool in TOOLS_DEFINITION:
        registry.register_tool(
            server_name="top",
            tool_name=tool["name"],
            description=tool["description"],
            input_schema=tool["input_schema"],
            handler=tool["handler"],
        )

    return registry
