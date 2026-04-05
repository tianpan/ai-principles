# -*- coding: utf-8 -*-
"""Knowledge MCP Server - 港华知识库 MCP 服务器。

本模块实现港华燃气知识库（Towngas Knowledge Base）的 MCP 服务器，
提供业务知识检索、FAQ 查询、应急处置指南、用气量预测等工具。

提供工具：
    - search_faq: 搜索常见问题，支持按关键词和分类查询
    - get_emergency_guide: 获取应急处置指南，包含处理步骤和联系方式
    - search_knowledge: 搜索知识库，获取设备和安全相关知识
    - get_gas_prediction: 获取用气量预测，基于历史数据和因素分析

数据说明：
    当前使用 Mock 数据实现，实际部署时应替换为：
    - FAQ_DATA: 对接客服知识库 API
    - EMERGENCY_GUIDES: 对接应急预案管理系统
    - KNOWLEDGE_BASE: 对接企业知识图谱或文档库

Example:
    注册 Knowledge 服务器到注册表::

        from app.mcp.registry import MCPToolRegistry
        from app.mcp.servers.knowledge_server import create_knowledge_server

        registry = MCPToolRegistry()
        create_knowledge_server(registry)

        # 使用工具
        result = await registry.execute_tool(
            "mcp__knowledge__search_faq",
            {"query": "燃气泄漏"}
        )
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from ..registry import MCPToolRegistry, MCPServerConfig


# ==================== 知识库数据 ====================

FAQ_DATA: List[Dict[str, Any]] = [
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
    """搜索常见问题（FAQ）。

    从 FAQ 知识库中搜索匹配的问答条目，支持按关键词和分类进行过滤。
    关键词匹配范围包括：问题标题、答案内容、关键词标签。

    Args:
        query: 搜索关键词，用于匹配问题、答案或关键词标签。
            不区分大小写，为空时不进行关键词过滤。
        category: 问题分类，用于限定搜索范围。
            可选值：用气安全、缴费服务、设备维护等。
            为空时不进行分类过滤。

    Returns:
        包含搜索结果的字典，结构如下：
            - success (bool): 搜索是否成功
            - data (list): 匹配的 FAQ 条目列表，每条包含
                id、category、question、answer、keywords 字段
            - count (int): 匹配结果数量
            - query (str): 使用的搜索关键词
            - category (str): 使用的分类过滤

    Example:
        按关键词搜索::\n

            result = await search_faq(query="泄漏")
            # 返回所有包含"泄漏"的 FAQ

        按分类搜索::\n

            result = await search_faq(category="用气安全")
            # 返回"用气安全"分类下的所有 FAQ

        组合搜索::\n

            result = await search_faq(query="缴费", category="缴费服务")
            # 返回"缴费服务"分类下包含"缴费"的 FAQ
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
    """获取应急处置指南。

    根据应急类型返回对应的处置步骤和联系信息。
    应急指南包含详细的处理步骤、严重等级和紧急联系方式。

    Args:
        emergency_type: 应急类型名称，支持模糊匹配。
            常见类型：
            - "管道泄漏": 管道破裂或燃气泄漏
            - "设备故障": 场站设备运行异常
            - "压力异常": 管网压力过高或过低

    Returns:
        包含应急指南的字典，结构如下：
            成功匹配时：
            - success (bool): True
            - data (dict): 应急指南详情
                - id (str): 指南 ID
                - type (str): 应急类型
                - severity (str): 严重程度（高/中/低）
                - steps (list): 处置步骤列表
                - contact (str): 紧急联系方式
            - timestamp (str): 查询时间戳

            未精确匹配时：
            - success (bool): True
            - data (dict): 包含提示信息和所有可用指南

    Example:
        获取管道泄漏应急指南::\n

            result = await get_emergency_guide("管道泄漏")
            # 返回管道泄漏的处置步骤和联系方式

        查询不存在的类型::\n

            result = await get_emergency_guide("未知故障")
            # 返回所有可用应急类型的列表
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
    """搜索业务知识库。

    从知识库中搜索相关的业务知识条目，包括设备原理、
    安全规范、操作规程等内容。

    Args:
        query: 搜索关键词，必填参数。用于匹配知识条目的
            标题或内容，不区分大小写。
        category: 知识分类，可选参数。用于限定搜索范围。
            可选值：
            - "设备知识": 设备原理、操作维护
            - "基础知识": 燃气特性、安全常识

    Returns:
        包含搜索结果的字典，结构如下：
            - success (bool): 搜索是否成功
            - data (list): 匹配的知识条目列表，每条包含
                id、title、content、category 字段
            - count (int): 匹配结果数量
            - query (str): 使用的搜索关键词
            - category (str): 使用的分类过滤

    Example:
        搜索调压器相关知识::\n

            result = await search_knowledge("调压")
            # 返回包含"调压"的知识条目

        搜索特定分类::\n

            result = await search_knowledge("天然气", category="基础知识")
            # 返回"基础知识"分类下关于天然气的条目
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
    """获取用气量预测。

    基于历史数据和影响因素（星期、温度等）预测指定日期的用气量。
    当前使用简化的 Mock 算法，实际部署应替换为 ML 模型。

    Args:
        date: 预测日期，格式为 YYYY-MM-DD。为空时默认预测明天。
            日期必须在合理范围内，格式错误会返回失败。
        station_id: 场站 ID，可选参数。用于获取特定场站的预测。
            为空时返回全区域的预测数据。

    Returns:
        包含预测结果的字典，结构如下：
            成功时：
            - success (bool): True
            - data (dict): 预测数据详情
                - date (str): 预测日期
                - station_id (str): 场站 ID 或 "all"
                - predicted_flow_m3 (int): 预测用气量（立方米）
                - confidence (str): 预测置信度（low/medium/high）
                - factors (dict): 影响因子
                    - weekday_factor (float): 周末调整系数
                    - temperature_factor (float): 温度调整系数
                - hourly_breakdown (list): 24小时分布预测
                - generated_at (str): 生成时间戳

            日期格式错误时：
            - success (bool): False
            - error (str): 错误信息

    Example:
        预测明天的用气量::\n

            result = await get_gas_prediction()
            # 返回明天的全区域用气量预测

        预测特定日期和场站::\n

            result = await get_gas_prediction(
                date="2024-03-15",
                station_id="ST001"
            )
            # 返回指定日期和场站的用气量预测
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

TOOLS_DEFINITION: List[Dict[str, Any]] = [
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
    """创建 Knowledge MCP Server 并注册到 Registry。

    创建知识库 MCP 服务器实例，将所有知识库工具注册到指定的注册表中。
    注册的工具包括：search_faq、get_emergency_guide、search_knowledge、
    get_gas_prediction。

    Args:
        registry: MCP 工具注册表实例，用于注册服务器和工具。
            通常由应用程序在初始化时创建和传入。

    Returns:
        注册了知识库工具的 Registry 实例，与传入的 registry 是同一对象，
        便于链式调用。

    Example:
        创建并注册服务器::\n

            from app.mcp.registry import MCPToolRegistry

            registry = MCPToolRegistry()
            create_knowledge_server(registry)

            # 获取注册的工具列表
            tools = registry.get_claude_tools()
            print(f"已注册 {len(tools)} 个知识库工具")

        与其他服务器一起注册::\n

            registry = MCPToolRegistry()
            create_top_server(registry)
            create_knowledge_server(registry)
            # 现在 registry 包含 TOP 和 Knowledge 两个服务器的工具
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
