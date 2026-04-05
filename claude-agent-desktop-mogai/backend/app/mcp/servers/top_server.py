# -*- coding: utf-8 -*-
"""TOP MCP Server - 港华运营平台 MCP 服务器。

本模块实现港华运营平台（Towngas Operation Platform）的 MCP 服务器，
提供场站管理、设备监控、管网运维等核心业务工具。

提供工具：
    - query_station: 查询场站信息，支持按 ID 或名称查询
    - query_device: 查询设备信息，包括状态和维护记录
    - get_pipeline_status: 获取管网运行状态
    - get_realtime_metrics: 获取场站实时监控指标
    - generate_daily_report: 生成运营日报

数据说明：
    当前使用 Mock 数据实现，实际部署时应替换为 TOP API 调用。
    Mock 数据包含深圳港华燃气的典型场站、设备和管网配置。

Example:
    注册 TOP 服务器到注册表::

        from app.mcp.registry import MCPToolRegistry
        from app.mcp.servers.top_server import create_top_server

        registry = MCPToolRegistry()
        create_top_server(registry)

        # 使用工具
        result = await registry.execute_tool(
            "mcp__top__query_station",
            {"station_id": "ST001"}
        )
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import random

from ..registry import MCPToolRegistry, MCPServerConfig


# ==================== 工具处理器 ====================

async def query_station(station_id: Optional[str] = None, station_name: Optional[str] = None) -> Dict[str, Any]:
    """查询场站信息。

    从运营平台查询场站的详细信息，支持按 ID 精确查询或按名称模糊匹配。
    如果不提供任何参数，返回所有场站列表。

    Args:
        station_id: 场站唯一标识符（如 "ST001"）。提供时进行精确匹配，
            与 station_name 互斥，优先使用 station_id。
        station_name: 场站名称（如 "龙岗"）。提供时进行模糊匹配，
            返回名称中包含该字符串的所有场站。

    Returns:
        包含查询结果的字典，结构如下：
            按 ID 查询成功时：
            - success (bool): True
            - data (dict): 单个场站信息，包含 id、name、type、status、
                address、pressure_in、pressure_out、flow_rate、last_update 字段

            按名称查询时：
            - success (bool): True
            - data (list): 匹配的场站列表
            - count (int): 匹配数量

            查询所有场站时：
            - success (bool): True
            - data (list): 所有场站列表
            - count (int): 场站总数

            查询失败时：
            - success (bool): False
            - error (str): 错误信息

    Example:
        按 ID 查询::\n

            result = await query_station(station_id="ST001")
            # 返回 ID 为 ST001 的场站信息

        按名称模糊查询::\n

            result = await query_station(station_name="龙岗")
            # 返回名称中包含"龙岗"的所有场站

        查询所有场站::\n

            result = await query_station()
            # 返回所有场站列表
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
    """查询设备信息。

    根据设备 ID 查询设备的详细信息，包括设备类型、运行状态、
    维护记录和技术参数等。

    Args:
        device_id: 设备唯一标识符（如 "DEV001"），必填参数。
            常见设备前缀：DEV（通用设备）、REG（调压器）、
            FLOW（流量计）等。

    Returns:
        包含查询结果的字典，结构如下：
            查询成功时：
            - success (bool): True
            - data (dict): 设备详情，包含以下字段：
                - id (str): 设备 ID
                - name (str): 设备名称
                - type (str): 设备类型（调压器、流量计等）
                - station_id (str): 所属场站 ID
                - status (str): 运行状态
                - manufacturer (str): 制造商
                - install_date (str): 安装日期
                - last_maintenance (str): 上次维护日期
                - next_maintenance (str): 下次维护日期
                - parameters (dict): 设备技术参数

            查询失败时：
            - success (bool): False
            - error (str): 错误信息（如 "未找到设备: XXX"）

    Example:
        查询调压器信息::\n

            result = await query_device("DEV001")
            if result["success"]:
                device = result["data"]
                print(f"设备名称: {device['name']}")
                print(f"运行状态: {device['status']}")
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
    """获取管网运行状态。

    查询燃气管网的运行状态信息，包括压力、流量、泄漏检测等。
    可查询指定管网或获取所有管网的概览。

    Args:
        pipeline_id: 管网唯一标识符（如 "PL001"），可选参数。
            不提供时返回所有管网的状态列表。

    Returns:
        包含查询结果的字典，结构如下：
            查询单个管网成功时：
            - success (bool): True
            - data (dict): 单个管网信息，包含以下字段：
                - id (str): 管网 ID
                - name (str): 管网名称
                - diameter (str): 管径（如 "DN300"）
                - material (str): 材质（PE、钢管等）
                - length_km (float): 长度（公里）
                - pressure (float): 运行压力（MPa）
                - flow_rate (int): 流量（m³/h）
                - status (str): 运行状态
                - leak_detection (str): 泄漏检测结果

            查询所有管网时：
            - success (bool): True
            - data (list): 所有管网状态列表
            - count (int): 管网数量

            查询失败时：
            - success (bool): False
            - error (str): 错误信息

    Example:
        查询指定管网::\n

            result = await get_pipeline_status("PL001")
            # 返回福华主管道的状态信息

        查询所有管网::\n

            result = await get_pipeline_status()
            for pipeline in result["data"]:
                print(f"{pipeline['name']}: {pipeline['status']}")
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
    """获取场站实时监控指标。

    从 SCADA 系统获取指定场站的实时运行数据，包括压力、温度、
    流量、阀门位置等关键指标。当前使用 Mock 数据模拟实时变化。

    Args:
        station_id: 场站唯一标识符（如 "ST001"），必填参数。

    Returns:
        包含实时监控数据的字典，结构如下：
            - success (bool): True（此接口总是返回成功）
            - data (dict): 实时数据详情
                - station_id (str): 场站 ID
                - timestamp (str): 数据时间戳（ISO 8601 格式）
                - metrics (dict): 监控指标
                    - pressure_in (float): 进口压力（MPa）
                    - pressure_out (float): 出口压力（MPa）
                    - temperature (float): 温度（摄氏度）
                    - flow_rate (int): 流量（m³/h）
                    - valve_position (int): 阀门开度（%）
                - alerts (list): 告警列表（空列表表示无告警）
                - status (str): 运行状态

    Example:
        获取实时数据::\n

            result = await get_realtime_metrics("ST001")
            metrics = result["data"]["metrics"]
            print(f"进口压力: {metrics['pressure_in']} MPa")
            print(f"流量: {metrics['flow_rate']} m³/h")
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
    """生成运营日报。

    汇总指定日期的运营数据，生成包含各场站流量、状态的运营日报。
    报告包括总览统计和各场站明细。

    Args:
        date: 报告日期，格式为 YYYY-MM-DD（如 "2024-03-15"）。
            不提供时默认使用今天的日期。

    Returns:
        包含日报数据的字典，结构如下：
            - success (bool): True（此接口总是返回成功）
            - data (dict): 日报详情
                - date (str): 报告日期
                - generated_at (str): 生成时间戳
                - summary (dict): 汇总统计
                    - total_stations (int): 场站总数
                    - active_stations (int): 正常运行场站数
                    - maintenance_stations (int): 维护中场站数
                    - total_flow_m3 (int): 总流量（立方米）
                    - avg_pressure (float): 平均压力（MPa）
                    - incidents (int): 事件数量
                - details (list): 各场站明细
                    - station (str): 场站名称
                    - flow_m3 (int): 流量（立方米）
                    - status (str): 运行状态

    Example:
        生成今日日报::\n

            result = await generate_daily_report()
            summary = result["data"]["summary"]
            print(f"总流量: {summary['total_flow_m3']} m³")

        生成指定日期日报::\n

            result = await generate_daily_report("2024-03-15")
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

TOOLS_DEFINITION: List[Dict[str, Any]] = [
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
    """创建 TOP MCP Server 并注册到 Registry。

    创建港华运营平台 MCP 服务器实例，将所有运营管理工具注册到指定的注册表中。
    注册的工具包括：query_station、query_device、get_pipeline_status、
    get_realtime_metrics、generate_daily_report。

    Args:
        registry: MCP 工具注册表实例，用于注册服务器和工具。
            通常由应用程序在初始化时创建和传入。

    Returns:
        注册了 TOP 工具的 Registry 实例，与传入的 registry 是同一对象，
        便于链式调用。

    Example:
        创建并注册服务器::\n

            from app.mcp.registry import MCPToolRegistry

            registry = MCPToolRegistry()
            create_top_server(registry)

            # 获取注册的工具列表
            tools = registry.get_claude_tools()
            print(f"已注册 {len(tools)} 个 TOP 工具")

        与其他服务器一起注册::\n

            registry = MCPToolRegistry()
            create_top_server(registry)
            create_knowledge_server(registry)
            # 现在 registry 包含 TOP 和 Knowledge 两个服务器的工具
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
