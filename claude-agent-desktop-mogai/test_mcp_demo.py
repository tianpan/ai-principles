# -*- coding: utf-8 -*-
"""
MCP 集成演示脚本

直接测试 MCP 工具执行
"""

import asyncio
import sys
import os

# 添加 backend 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.mcp import MCPToolRegistry
from app.mcp.servers import create_top_server, create_knowledge_server


async def demo():
    """演示 MCP 工具功能"""
    print("=" * 60)
    print("  MCP 工具集成演示 - Towngas Manus")
    print("=" * 60)
    print()

    # 创建 MCP Registry 并注册服务器
    registry = MCPToolRegistry()
    create_top_server(registry)
    create_knowledge_server(registry)

    # 显示统计信息
    stats = registry.get_stats()
    print(f"📊 注册统计:")
    print(f"   - 服务器数量: {stats['total_servers']}")
    print(f"   - 工具总数: {stats['total_tools']}")
    for server, count in stats['tools_by_server'].items():
        print(f"   - {server}: {count} 个工具")
    print()

    # 列出所有 MCP 工具
    print("🔧 已注册的 MCP 工具:")
    for tool in registry.get_all_tools():
        print(f"   [{tool.server_name}] {tool.mcp_name}")
        print(f"       {tool.description[:50]}...")
    print()

    # 演示工具执行
    demos = [
        {
            "name": "mcp__top__query_station",
            "args": {"station_id": "ST001"},
            "desc": "查询场站信息"
        },
        {
            "name": "mcp__top__get_weather",
            "args": {"city": "深圳"},
            "desc": "获取天气信息"
        },
        {
            "name": "mcp__knowledge__search_faq",
            "args": {"query": "燃气泄漏"},
            "desc": "搜索 FAQ"
        },
        {
            "name": "mcp__knowledge__get_emergency_guide",
            "args": {"emergency_type": "管道泄漏"},
            "desc": "获取应急指南"
        },
        {
            "name": "mcp__top__generate_daily_report",
            "args": {},
            "desc": "生成日报"
        },
    ]

    for i, demo in enumerate(demos, 1):
        print(f"{'=' * 60}")
        print(f"  演示 {i}: {demo['desc']}")
        print(f"  工具: {demo['name']}")
        print(f"  参数: {demo['args']}")
        print("-" * 60)

        result = await registry.execute_tool(demo['name'], demo['args'])

        if result.success:
            print("  ✅ 执行成功")
            print(f"  ⏱️  耗时: {result.execution_time_ms:.2f}ms")
            print("  📦 结果:")

            # 格式化输出
            data = result.result
            if isinstance(data, dict):
                for key, value in list(data.items())[:5]:  # 限制显示5个字段
                    if isinstance(value, (list, dict)):
                        print(f"     - {key}: {type(value).__name__} ({len(value)} 项)")
                    else:
                        val_str = str(value)[:60]
                        print(f"     - {key}: {val_str}")
            else:
                print(f"     {data}")
        else:
            print(f"  ❌ 执行失败: {result.error}")

        print()

    print("=" * 60)
    print("  演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
