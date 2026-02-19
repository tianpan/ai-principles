# -*- coding: utf-8 -*-
"""
Skills Registry - 技能注册系统

负责技能的管理，包括：
- 技能注册、发现
- 技能执行
- 内置技能示例
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class SkillParameter:
    """
    技能参数定义

    Attributes:
        name: 参数名称
        type: 参数类型
        description: 参数描述
        required: 是否必填
        default: 默认值
    """
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass
class Skill:
    """
    技能数据类

    Attributes:
        name: 技能名称（唯一标识）
        description: 技能描述
        parameters: 参数 JSON Schema
        handler: 执行函数
        category: 技能分类
        metadata: 额外元数据
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式（不包含 handler）

        Returns:
            技能信息字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
            "metadata": self.metadata
        }


class BaseSkill(ABC):
    """
    技能基类

    继承此类来创建自定义技能
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """技能名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """技能描述"""
        pass

    @property
    def category(self) -> str:
        """技能分类"""
        return "general"

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        参数 JSON Schema

        返回 Claude API 兼容的参数定义
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        执行技能

        Args:
            **kwargs: 技能参数

        Returns:
            执行结果
        """
        pass

    def to_skill(self) -> Skill:
        """
        转换为 Skill 对象

        Returns:
            Skill 实例
        """
        return Skill(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            handler=self.execute,
            category=self.category
        )


# ==================== 内置技能示例 ====================

class GetCurrentTimeSkill(BaseSkill):
    """获取当前时间技能"""

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return "获取当前的日期和时间"

    @property
    def category(self) -> str:
        return "utility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区，如 'Asia/Shanghai'"
                }
            },
            "required": []
        }

    async def execute(self, timezone: str = "Asia/Shanghai") -> Dict[str, str]:
        """
        执行获取时间

        Args:
            timezone: 时区

        Returns:
            包含当前时间信息的字典
        """
        now = datetime.now()
        return {
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
            "timezone": timezone
        }


class CalculatorSkill(BaseSkill):
    """计算器技能"""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "执行数学计算，支持基本运算和常用函数"

    @property
    def category(self) -> str:
        return "utility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'"
                }
            },
            "required": ["expression"]
        }

    async def execute(self, expression: str) -> Dict[str, Any]:
        """
        执行计算

        Args:
            expression: 数学表达式

        Returns:
            包含计算结果的字典
        """
        try:
            # 使用 simpleeval 库进行安全的表达式求值
            from simpleeval import simple_eval, EvalWithCompoundTypes

            # 创建安全的计算环境
            evaluator = EvalWithCompoundTypes()

            # 添加安全的数学函数
            import math
            evaluator.functions.update({
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "log10": math.log10,
                "exp": math.exp,
                "floor": math.floor,
                "ceil": math.ceil,
            })

            # 添加数学常量
            evaluator.names.update({
                "pi": math.pi,
                "e": math.e,
            })

            # 使用 simple_eval 进行安全求值
            result = evaluator.eval(expression)
            return {
                "expression": expression,
                "result": result,
                "success": True
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "success": False
            }


class EchoSkill(BaseSkill):
    """回声技能 - 用于测试"""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "回显用户输入的消息，用于测试"

    @property
    def category(self) -> str:
        return "utility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "要回显的消息"
                }
            },
            "required": ["message"]
        }

    async def execute(self, message: str) -> Dict[str, str]:
        """
        执行回显

        Args:
            message: 消息内容

        Returns:
            包含回显消息的字典
        """
        return {
            "echo": message,
            "timestamp": datetime.now().isoformat()
        }


class MockStationQuerySkill(BaseSkill):
    """
    模拟场站查询技能

    这是一个示例技能，模拟从 TOP 系统查询场站数据
    实际部署时需要替换为真实的 TOP API 调用
    """

    @property
    def name(self) -> str:
        return "query_station"

    @property
    def description(self) -> str:
        return "查询场站基本信息、运行状态等数据（模拟）"

    @property
    def category(self) -> str:
        return "top"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "station_id": {
                    "type": "string",
                    "description": "场站 ID 或名称"
                },
                "info_type": {
                    "type": "string",
                    "enum": ["basic", "status", "devices", "all"],
                    "description": "信息类型：basic-基本信息, status-运行状态, devices-设备列表, all-全部"
                }
            },
            "required": ["station_id"]
        }

    async def execute(
        self,
        station_id: str,
        info_type: str = "basic"
    ) -> Dict[str, Any]:
        """
        执行场站查询（模拟）

        Args:
            station_id: 场站 ID 或名称
            info_type: 信息类型

        Returns:
            场站信息字典
        """
        # 模拟数据
        mock_data = {
            "station_001": {
                "id": "station_001",
                "name": "城西门站",
                "type": "门站",
                "location": "XX市XX区XX路100号",
                "status": "正常运行",
                "daily_gas_volume": "125000 m³",
                "pressure": "0.45 MPa",
                "devices": [
                    {"name": "调压器A", "status": "正常"},
                    {"name": "调压器B", "status": "正常"},
                    {"name": "流量计", "status": "正常"}
                ]
            },
            "station_002": {
                "id": "station_002",
                "name": "东郊调压站",
                "type": "调压站",
                "location": "XX市XX区XX路200号",
                "status": "正常运行",
                "daily_gas_volume": "45000 m³",
                "pressure": "0.28 MPa",
                "devices": [
                    {"name": "调压器", "status": "正常"},
                    {"name": "流量计", "status": "正常"}
                ]
            }
        }

        # 查找场站
        station = None
        for sid, data in mock_data.items():
            if sid == station_id or data["name"] == station_id:
                station = data
                break

        if not station:
            return {
                "success": False,
                "error": f"未找到场站: {station_id}",
                "available_stations": list(mock_data.keys())
            }

        # 根据信息类型返回
        if info_type == "basic":
            return {
                "success": True,
                "data": {
                    "id": station["id"],
                    "name": station["name"],
                    "type": station["type"],
                    "location": station["location"]
                }
            }
        elif info_type == "status":
            return {
                "success": True,
                "data": {
                    "id": station["id"],
                    "name": station["name"],
                    "status": station["status"],
                    "daily_gas_volume": station["daily_gas_volume"],
                    "pressure": station["pressure"]
                }
            }
        elif info_type == "devices":
            return {
                "success": True,
                "data": {
                    "id": station["id"],
                    "name": station["name"],
                    "devices": station["devices"]
                }
            }
        else:  # all
            return {
                "success": True,
                "data": station,
                "note": "这是模拟数据，实际部署时将连接 TOP API 获取真实数据"
            }


class SkillsRegistry:
    """
    技能注册表

    负责技能的注册、发现和执行

    Attributes:
        _skills: 技能字典
        _categories: 技能分类字典
    """

    def __init__(self):
        """初始化技能注册表"""
        self._skills: Dict[str, Skill] = {}
        self._categories: Dict[str, List[str]] = {}

        # 注册内置技能
        self._register_builtin_skills()

    def _register_builtin_skills(self) -> None:
        """注册内置技能"""
        builtin_skills = [
            GetCurrentTimeSkill(),
            CalculatorSkill(),
            EchoSkill(),
            MockStationQuerySkill()
        ]

        for skill in builtin_skills:
            self.register_skill(skill.to_skill())

    def register_skill(self, skill: Skill) -> bool:
        """
        注册技能

        Args:
            skill: 要注册的技能

        Returns:
            注册成功返回 True，如果技能已存在返回 False
        """
        if skill.name in self._skills:
            return False

        self._skills[skill.name] = skill

        # 更新分类索引
        if skill.category not in self._categories:
            self._categories[skill.category] = []
        self._categories[skill.category].append(skill.name)

        return True

    def unregister_skill(self, name: str) -> bool:
        """
        注销技能

        Args:
            name: 技能名称

        Returns:
            注销成功返回 True，如果技能不存在返回 False
        """
        if name not in self._skills:
            return False

        skill = self._skills[name]
        del self._skills[name]

        # 更新分类索引
        if skill.category in self._categories:
            self._categories[skill.category].remove(name)

        return True

    def get_skill(self, name: str) -> Optional[Skill]:
        """
        获取技能

        Args:
            name: 技能名称

        Returns:
            Skill 实例，如果不存在则返回 None
        """
        return self._skills.get(name)

    def get_all_skills(self) -> List[Skill]:
        """
        获取所有技能

        Returns:
            Skill 列表
        """
        return list(self._skills.values())

    def get_skills_by_category(self, category: str) -> List[Skill]:
        """
        按分类获取技能

        Args:
            category: 分类名称

        Returns:
            该分类下的 Skill 列表
        """
        skill_names = self._categories.get(category, [])
        return [self._skills[name] for name in skill_names if name in self._skills]

    def get_categories(self) -> List[str]:
        """
        获取所有分类

        Returns:
            分类名称列表
        """
        return list(self._categories.keys())

    async def execute_skill(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        执行技能

        Args:
            name: 技能名称
            arguments: 技能参数

        Returns:
            执行结果

        Raises:
            ValueError: 技能不存在
        """
        skill = self.get_skill(name)
        if not skill:
            raise ValueError(f"技能不存在: {name}")

        # 调用技能处理函数
        if asyncio.iscoroutinefunction(skill.handler):
            result = await skill.handler(**arguments)
        else:
            result = skill.handler(**arguments)

        return result

    def get_skill_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取技能信息

        Args:
            name: 技能名称

        Returns:
            技能信息字典
        """
        skill = self.get_skill(name)
        if not skill:
            return None

        return skill.to_dict()

    def get_all_skills_info(self) -> List[Dict[str, Any]]:
        """
        获取所有技能信息

        Returns:
            技能信息列表
        """
        return [skill.to_dict() for skill in self.get_all_skills()]
