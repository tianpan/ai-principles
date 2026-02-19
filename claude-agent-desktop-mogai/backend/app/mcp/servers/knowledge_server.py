# -*- coding: utf-8 -*-
"""
Knowledge MCP Server - 知识库 MCP 服务器

提供港华业务知识检索、FAQ 查询、应急处置指南等工具
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from ..registry import MCPToolRegistry, MCPServerConfig


# ==================== 知识库数据 ====================

FAQ_DATA = [
    {
        "id": "FAQ001",
        "category": "用气安全",
        "question": "发现燃气泄漏应该怎么办？",
        "answer": """
1. 立即关闭燃气阀门
2. 打开门窗通风，切勿开关电器
3. 禁止使用明火或电话
4. 撤离到安全区域后拨打港华燃气服务热线
5. 等待专业人员处理
""",
        "keywords": ["泄漏", "安全", "应急"],
    },
    {
        "id": "FAQ002",
        "category": "缴费服务",
        "question": "如何查询和缴纳燃气费？",
        "answer": """
您可以通过以下方式查询和缴纳燃气费：
1. 港华燃气微信公众号 - 在线缴费
2. 支付宝/微信 - 生活缴费
3. 银行代扣 - 自动扣款
4. 营业厅现金缴费
5. 自助终端机缴费
""",
        "keywords": ["缴费", "费用", "支付"],
    },
    {
        "id": "FAQ003",
        "category": "设备维护",
        "question": "燃气表需要多久检定一次？",
        "answer": """
根据国家规定：
- 家用燃气表使用期限一般不超过 10 年
- 使用中的燃气表需要定期检定
- 如发现计量异常，可申请免费检定
- 联系客服热线预约上门检定服务
""",
        "keywords": ["燃气表", "检定", "维护"],
    },
]

EMERGENCY_GUIDES = [
    {
        "id": "EM001",
        "type": "管道泄漏",
        "severity": "高",
        "steps": [
            "立即疏散周边人员，设置警戒区",
            "关闭上下游阀门，切断气源",
            "通知应急抢修队伍",
            "使用可燃气体检测仪监测浓度",
            "确保现场无明火、无火花源",
            "抢修完成后进行气密性测试",
        ],
        "contact": "24小时应急热线: 95777",
    },
    {
        "id": "EM002",
        "type": "设备故障",
        "severity": "中",
        "steps": [
            "确认故障设备类型和位置",
            "评估对供气的影响范围",
            "启动备用设备（如有）",
            "通知维修人员到场",
            "记录故障现象和处理过程",
            "恢复后进行功能测试",
        ],
        "contact": "运维中心: 0755-88888888",
    },
    {
        "id": "EM003",
        "type": "压力异常",
        "severity": "高",
        "steps": [
            "监控压力变化趋势",
            "判断是过高还是过低",
            "过高：检查调压器，必要时放散",
            "过低：检查气源和管道堵塞",
            "通知调度中心调整供气计划",
            "持续监控直至恢复正常",
        ],
        "contact": "调度中心: 0755-88888889",
    },
]

KNOWLEDGE_BASE = [
    {
        "id": "KB001",
        "title": "燃气调压站工作原理",
        "content": """
调压站是燃气输配系统的重要设施，主要功能包括：
1. 降低燃气压力：将高压燃气降至用户所需压力
2. 稳定压力输出：确保下游压力稳定
3. 安全保护：超压切断、放散保护
4. 计量功能：部分调压站配有计量设备

主要设备：
- 调压器：核心设备，调节压力
- 切断阀：超压时自动切断
- 放散阀：超压时安全放散
- 过滤器：过滤杂质
- 压力表：监测压力
""",
        "category": "设备知识",
    },
    {
        "id": "KB002",
        "title": "天然气特性",
        "content": """
天然气主要特性：
1. 主要成分：甲烷（CH4）约 90% 以上
2. 密度：约 0.7-0.8 kg/m³（比空气轻）
3. 爆炸极限：5% - 15%（体积比）
4. 热值：约 36-40 MJ/m³
5. 燃烧特性：蓝色火焰，充分燃烧产物为 CO2 和 H2O

安全特性：
- 无色无味（添加臭剂后有大蒜味）
- 易燃易爆
- 窒息性（高浓度时）
""",
        "category": "基础知识",
    },
]


# ==================== 工具处理器 ====================

async def search_faq(
    query: Optional[str] = None,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    搜索常见问题

    Args:
        query: 搜索关键词
        category: 问题分类

    Returns:
        匹配的 FAQ 列表
    """
    results = FAQ_DATA

    # 按分类过滤
    if category:
        results = [f for f in results if f["category"] == category]

    # 按关键词搜索
    if query:
        query_lower = query.lower()
        results = [
            f for f in results
            if query_lower in f["question"].lower()
            or query_lower in f["answer"].lower()
            or any(query_lower in kw for kw in f.get("keywords", []))
        ]

    return {
        "success": True,
        "data": results,
        "count": len(results),
        "query": query,
        "category": category,
    }


async def get_emergency_guide(emergency_type: str) -> Dict[str, Any]:
    """
    获取应急处置指南

    Args:
        emergency_type: 应急类型（管道泄漏、设备故障、压力异常等）

    Returns:
        应急处置步骤
    """
    for guide in EMERGENCY_GUIDES:
        if emergency_type in guide["type"]:
            return {
                "success": True,
                "data": guide,
                "timestamp": datetime.now().isoformat(),
            }

    # 如果没有精确匹配，返回所有应急指南列表
    return {
        "success": True,
        "data": {
            "message": f"未找到 '{emergency_type}' 的精确匹配，以下是所有可用应急指南",
            "available_types": [g["type"] for g in EMERGENCY_GUIDES],
            "all_guides": EMERGENCY_GUIDES,
        },
    }


async def search_knowledge(
    query: str,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    搜索知识库

    Args:
        query: 搜索关键词
        category: 知识分类（可选）

    Returns:
        匹配的知识条目
    """
    results = KNOWLEDGE_BASE

    # 按分类过滤
    if category:
        results = [k for k in results if k["category"] == category]

    # 按关键词搜索
    if query:
        query_lower = query.lower()
        results = [
            k for k in results
            if query_lower in k["title"].lower()
            or query_lower in k["content"].lower()
        ]

    return {
        "success": True,
        "data": results,
        "count": len(results),
        "query": query,
        "category": category,
    }


async def get_gas_prediction(
    date: Optional[str] = None,
    station_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取用气量预测（基于历史数据的简单预测）

    Args:
        date: 预测日期（YYYY-MM-DD）
        station_id: 场站 ID（可选）

    Returns:
        用气量预测数据
    """
    # 设置默认日期为明天
    target_date = date or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # 验证日期格式
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        return {
            "success": False,
            "error": f"无效的日期格式: {target_date}，期望格式: YYYY-MM-DD",
        }

    # Mock 预测数据 - 实际应使用 ML 模型
    base_flow = 45000  # 基础用气量 m³

    # 根据日期类型调整（周末较低）
    weekday_factor = 0.9 if dt.weekday() >= 5 else 1.0

    # 温度影响（简化处理）
    temperature_factor = random.uniform(0.95, 1.05)

    predicted_flow = int(base_flow * weekday_factor * temperature_factor)

    return {
        "success": True,
        "data": {
            "date": target_date,
            "station_id": station_id or "all",
            "predicted_flow_m3": predicted_flow,
            "confidence": "medium",
            "factors": {
                "weekday_factor": weekday_factor,
                "temperature_factor": round(temperature_factor, 2),
            },
            "hourly_breakdown": [
                {"hour": h, "flow": int(predicted_flow / 24 * random.uniform(0.8, 1.2))}
                for h in range(24)
            ],
            "generated_at": datetime.now().isoformat(),
        },
    }


# ==================== 工具定义 ====================

TOOLS_DEFINITION = [
    {
        "name": "search_faq",
        "description": "搜索常见问题解答，支持按关键词和分类查询",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "category": {
                    "type": "string",
                    "description": "问题分类（用气安全、缴费服务、设备维护等）",
                },
            },
        },
        "handler": search_faq,
    },
    {
        "name": "get_emergency_guide",
        "description": "获取应急处置指南，包括处理步骤和联系方式",
        "input_schema": {
            "type": "object",
            "properties": {
                "emergency_type": {
                    "type": "string",
                    "description": "应急类型（管道泄漏、设备故障、压力异常等）",
                },
            },
            "required": ["emergency_type"],
        },
        "handler": get_emergency_guide,
    },
    {
        "name": "search_knowledge",
        "description": "搜索知识库，获取燃气设备、安全、操作等相关知识",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "category": {
                    "type": "string",
                    "description": "知识分类（设备知识、基础知识等）",
                },
            },
            "required": ["query"],
        },
        "handler": search_knowledge,
    },
    {
        "name": "get_gas_prediction",
        "description": "获取用气量预测，基于历史数据和天气因素",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "预测日期（YYYY-MM-DD），默认明天",
                },
                "station_id": {
                    "type": "string",
                    "description": "场站 ID（可选）",
                },
            },
        },
        "handler": get_gas_prediction,
    },
]


# ==================== 服务器创建 ====================

def create_knowledge_server(registry: MCPToolRegistry) -> MCPToolRegistry:
    """
    创建 Knowledge MCP Server 并注册到 Registry

    Args:
        registry: MCP 工具注册表

    Returns:
        注册了知识库工具的 Registry
    """
    # 注册服务器
    registry.register_server(MCPServerConfig(
        name="knowledge",
        type="local",
        description="知识库服务 - FAQ、应急指南、业务知识",
        enabled=True,
    ))

    # 注册工具
    for tool in TOOLS_DEFINITION:
        registry.register_tool(
            server_name="knowledge",
            tool_name=tool["name"],
            description=tool["description"],
            input_schema=tool["input_schema"],
            handler=tool["handler"],
        )

    return registry
