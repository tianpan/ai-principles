"""
Skills Registry 测试
====================

测试技能注册表功能：
- 技能注册
- 技能发现
- 技能执行
- 参数验证
- 技能分类

测试覆盖：
- 单元测试：独立技能操作
- 集成测试：技能执行流程
- 边界测试：异常参数处理
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Skills Registry 类（用于测试）
# =============================================================================


class MockSkill:
    """模拟技能类"""

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        category: str = "general",
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.handler = handler
        self.parameters = parameters or {}
        self.category = category
        self.tags = tags or []
        self._is_async = inspect.iscoroutinefunction(handler)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行技能"""
        if self._is_async:
            return await self.handler(params)
        return self.handler(params)

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """验证参数"""
        required = self.parameters.get("required", [])
        for param_name in required:
            if param_name not in params:
                raise ValueError(f"缺少必需参数: {param_name}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
            "tags": self.tags,
        }


class MockSkillsRegistry:
    """
    模拟技能注册表实现

    提供技能的注册、发现、执行和管理功能。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._skills: Dict[str, MockSkill] = {}
        self._categories: Dict[str, List[str]] = {}
        self._tags: Dict[str, List[str]] = {}

    def register(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        category: str = "general",
        tags: Optional[List[str]] = None,
    ) -> MockSkill:
        """
        注册技能

        Args:
            name: 技能名称
            handler: 处理函数
            description: 描述
            parameters: 参数定义
            category: 分类
            tags: 标签列表

        Returns:
            注册的技能对象
        """
        if name in self._skills:
            raise ValueError(f"技能 {name} 已存在")

        skill = MockSkill(
            name=name,
            description=description,
            handler=handler,
            parameters=parameters,
            category=category,
            tags=tags or [],
        )

        self._skills[name] = skill

        # 更新分类索引
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

        # 更新标签索引
        for tag in skill.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            self._tags[tag].append(name)

        return skill

    def unregister(self, name: str) -> bool:
        """
        注销技能

        Args:
            name: 技能名称

        Returns:
            是否成功注销
        """
        if name not in self._skills:
            return False

        skill = self._skills[name]

        # 从分类中移除
        if skill.category in self._categories:
            self._categories[skill.category].remove(name)
            if not self._categories[skill.category]:
                del self._categories[skill.category]

        # 从标签中移除
        for tag in skill.tags:
            if tag in self._tags:
                self._tags[tag].remove(name)
                if not self._tags[tag]:
                    del self._tags[tag]

        del self._skills[name]
        return True

    def get(self, name: str) -> Optional[MockSkill]:
        """
        获取技能

        Args:
            name: 技能名称

        Returns:
            技能对象，不存在返回 None
        """
        return self._skills.get(name)

    def list_all(self) -> List[Dict[str, Any]]:
        """
        列出所有技能

        Returns:
            技能列表
        """
        return [skill.to_dict() for skill in self._skills.values()]

    def list_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        按分类列出技能

        Args:
            category: 分类名称

        Returns:
            技能列表
        """
        skill_names = self._categories.get(category, [])
        return [self._skills[name].to_dict() for name in skill_names]

    def list_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        按标签列出技能

        Args:
            tag: 标签名称

        Returns:
            技能列表
        """
        skill_names = self._tags.get(tag, [])
        return [self._skills[name].to_dict() for name in skill_names]

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        搜索技能

        Args:
            query: 搜索关键词

        Returns:
            匹配的技能列表
        """
        query_lower = query.lower()
        results = []

        for skill in self._skills.values():
            # 在名称、描述、标签中搜索
            if (
                query_lower in skill.name.lower()
                or query_lower in skill.description.lower()
                or any(query_lower in tag.lower() for tag in skill.tags)
            ):
                results.append(skill.to_dict())

        return results

    async def execute(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行技能

        Args:
            name: 技能名称
            params: 执行参数

        Returns:
            执行结果
        """
        skill = self.get(name)
        if skill is None:
            raise ValueError(f"技能 {name} 不存在")

        # 验证参数
        skill.validate_params(params)

        # 执行技能
        try:
            result = await skill.execute(params)
            return {"success": True, "result": result, "skill": name}
        except Exception as e:
            return {"success": False, "error": str(e), "skill": name}

    def get_categories(self) -> List[str]:
        """获取所有分类"""
        return list(self._categories.keys())

    def get_tags(self) -> List[str]:
        """获取所有标签"""
        return list(self._tags.keys())

    def get_skill_count(self) -> int:
        """获取技能数量"""
        return len(self._skills)

    def clear(self) -> None:
        """清空所有技能"""
        self._skills.clear()
        self._categories.clear()
        self._tags.clear()


# =============================================================================
# 测试用技能处理器
# =============================================================================


def sync_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """同步处理器示例"""
    return {"message": f"同步处理完成: {params}"}


async def async_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """异步处理器示例"""
    await asyncio.sleep(0.01)
    return {"message": f"异步处理完成: {params}"}


def error_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """会抛出异常的处理器"""
    raise RuntimeError("处理器执行失败")


# =============================================================================
# 测试类
# =============================================================================


class TestSkillRegistration:
    """技能注册测试"""

    @pytest.fixture
    def registry(self):
        """创建技能注册表"""
        return MockSkillsRegistry()

    @pytest.mark.unit
    def test_register_skill_success(self, registry):
        """测试成功注册技能"""
        skill = registry.register(
            name="test_skill",
            handler=sync_handler,
            description="测试技能",
        )

        assert skill.name == "test_skill"
        assert skill.description == "测试技能"
        assert registry.get_skill_count() == 1

    @pytest.mark.unit
    def test_register_skill_with_parameters(self, registry):
        """测试带参数定义注册技能"""
        parameters = {
            "type": "object",
            "properties": {"region": {"type": "string"}},
            "required": ["region"],
        }

        skill = registry.register(
            name="price_query",
            handler=sync_handler,
            description="查询价格",
            parameters=parameters,
        )

        assert skill.parameters == parameters

    @pytest.mark.unit
    def test_register_skill_with_category(self, registry):
        """测试带分类注册技能"""
        skill = registry.register(
            name="inquiry_skill",
            handler=sync_handler,
            description="查询技能",
            category="inquiry",
        )

        assert skill.category == "inquiry"
        assert "inquiry" in registry.get_categories()

    @pytest.mark.unit
    def test_register_skill_with_tags(self, registry):
        """测试带标签注册技能"""
        skill = registry.register(
            name="tagged_skill",
            handler=sync_handler,
            description="带标签的技能",
            tags=["utility", "fast"],
        )

        assert "utility" in skill.tags
        assert "fast" in skill.tags
        assert "utility" in registry.get_tags()

    @pytest.mark.unit
    def test_register_duplicate_skill_raises_error(self, registry):
        """测试注册重复技能抛出异常"""
        registry.register("duplicate", sync_handler, "第一个")

        with pytest.raises(ValueError, match="已存在"):
            registry.register("duplicate", sync_handler, "第二个")

    @pytest.mark.unit
    def test_unregister_skill_success(self, registry):
        """测试成功注销技能"""
        registry.register("to_remove", sync_handler, "待删除")

        result = registry.unregister("to_remove")

        assert result is True
        assert registry.get("to_remove") is None

    @pytest.mark.unit
    def test_unregister_non_existent_skill(self, registry):
        """测试注销不存在的技能"""
        result = registry.unregister("non_existent")

        assert result is False

    @pytest.mark.unit
    def test_unregister_updates_categories(self, registry):
        """测试注销技能更新分类"""
        registry.register(
            "cat_skill",
            sync_handler,
            "分类技能",
            category="test_category",
        )

        registry.unregister("cat_skill")

        assert "test_category" not in registry.get_categories()


class TestSkillDiscovery:
    """技能发现测试"""

    @pytest.fixture
    def registry(self):
        """创建并填充技能注册表"""
        registry = MockSkillsRegistry()

        # 注册多个技能
        registry.register(
            "gas_price",
            sync_handler,
            "查询燃气价格",
            category="inquiry",
            tags=["gas", "price"],
        )
        registry.register(
            "bill_calc",
            sync_handler,
            "计算燃气账单",
            category="utility",
            tags=["gas", "bill", "calculator"],
        )
        registry.register(
            "appointment",
            sync_handler,
            "预约燃气服务",
            category="service",
            tags=["gas", "appointment"],
        )
        registry.register(
            "electric_price",
            sync_handler,
            "查询电价",
            category="inquiry",
            tags=["electric", "price"],
        )

        return registry

    @pytest.mark.unit
    def test_list_all_skills(self, registry):
        """测试列出所有技能"""
        skills = registry.list_all()

        assert len(skills) == 4

    @pytest.mark.unit
    def test_list_by_category(self, registry):
        """测试按分类列出技能"""
        inquiry_skills = registry.list_by_category("inquiry")

        assert len(inquiry_skills) == 2
        skill_names = [s["name"] for s in inquiry_skills]
        assert "gas_price" in skill_names
        assert "electric_price" in skill_names

    @pytest.mark.unit
    def test_list_by_non_existent_category(self, registry):
        """测试列出不存在分类的技能"""
        skills = registry.list_by_category("non_existent")

        assert skills == []

    @pytest.mark.unit
    def test_list_by_tag(self, registry):
        """测试按标签列出技能"""
        gas_skills = registry.list_by_tag("gas")

        assert len(gas_skills) == 3

    @pytest.mark.unit
    def test_search_by_name(self, registry):
        """测试按名称搜索"""
        results = registry.search("price")

        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "gas_price" in names
        assert "electric_price" in names

    @pytest.mark.unit
    def test_search_by_description(self, registry):
        """测试按描述搜索"""
        results = registry.search("预约")

        assert len(results) == 1
        assert results[0]["name"] == "appointment"

    @pytest.mark.unit
    def test_search_by_tag(self, registry):
        """测试按标签搜索"""
        results = registry.search("calculator")

        assert len(results) == 1
        assert results[0]["name"] == "bill_calc"

    @pytest.mark.unit
    def test_search_no_results(self, registry):
        """测试搜索无结果"""
        results = registry.search("不存在的关键词")

        assert results == []

    @pytest.mark.unit
    def test_search_case_insensitive(self, registry):
        """测试搜索不区分大小写"""
        results = registry.search("GAS")

        assert len(results) == 3

    @pytest.mark.unit
    def test_get_categories(self, registry):
        """测试获取所有分类"""
        categories = registry.get_categories()

        assert len(categories) == 3
        assert "inquiry" in categories
        assert "utility" in categories
        assert "service" in categories

    @pytest.mark.unit
    def test_get_tags(self, registry):
        """测试获取所有标签"""
        tags = registry.get_tags()

        assert len(tags) > 0
        assert "gas" in tags


class TestSkillExecution:
    """技能执行测试"""

    @pytest.fixture
    def registry(self):
        """创建技能注册表"""
        registry = MockSkillsRegistry()

        # 注册同步处理器
        registry.register(
            "sync_skill",
            sync_handler,
            "同步技能",
            parameters={"required": []},
        )

        # 注册异步处理器
        registry.register(
            "async_skill",
            async_handler,
            "异步技能",
            parameters={"required": []},
        )

        # 注册带参数验证的技能
        registry.register(
            "validated_skill",
            sync_handler,
            "验证技能",
            parameters={"required": ["region"]},
        )

        # 注册会抛出异常的技能
        registry.register(
            "error_skill",
            error_handler,
            "错误技能",
            parameters={"required": []},
        )

        return registry

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_sync_skill(self, registry):
        """测试执行同步技能"""
        result = await registry.execute("sync_skill", {"key": "value"})

        assert result["success"] is True
        assert "result" in result
        assert result["skill"] == "sync_skill"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_async_skill(self, registry):
        """测试执行异步技能"""
        result = await registry.execute("async_skill", {"key": "value"})

        assert result["success"] is True
        assert "异步处理完成" in result["result"]["message"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_skill_with_validation(self, registry):
        """测试带参数验证的技能执行"""
        # 正确参数
        result = await registry.execute("validated_skill", {"region": "香港"})
        assert result["success"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_skill_missing_required_param(self, registry):
        """测试缺少必需参数"""
        with pytest.raises(ValueError, match="缺少必需参数"):
            await registry.execute("validated_skill", {})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_non_existent_skill(self, registry):
        """测试执行不存在的技能"""
        with pytest.raises(ValueError, match="不存在"):
            await registry.execute("non_existent", {})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_skill_with_error(self, registry):
        """测试执行抛出异常的技能"""
        result = await registry.execute("error_skill", {})

        assert result["success"] is False
        assert "error" in result
        assert "处理器执行失败" in result["error"]


class TestSkillParameters:
    """技能参数测试"""

    @pytest.fixture
    def registry(self):
        """创建技能注册表"""
        return MockSkillsRegistry()

    @pytest.mark.unit
    def test_validate_required_params_success(self, registry):
        """测试验证必需参数成功"""
        registry.register(
            "test",
            sync_handler,
            parameters={"required": ["region", "type"]},
        )

        skill = registry.get("test")
        result = skill.validate_params({"region": "HK", "type": "residential"})

        assert result is True

    @pytest.mark.unit
    def test_validate_missing_required_param(self, registry):
        """测试缺少必需参数"""
        registry.register(
            "test",
            sync_handler,
            parameters={"required": ["region", "type"]},
        )

        skill = registry.get("test")

        with pytest.raises(ValueError, match="缺少必需参数"):
            skill.validate_params({"region": "HK"})  # 缺少 type

    @pytest.mark.unit
    def test_validate_no_required_params(self, registry):
        """测试无必需参数的验证"""
        registry.register(
            "test",
            sync_handler,
            parameters={"required": []},
        )

        skill = registry.get("test")
        result = skill.validate_params({})

        assert result is True

    @pytest.mark.unit
    def test_extra_params_allowed(self, registry):
        """测试额外参数被允许"""
        registry.register(
            "test",
            sync_handler,
            parameters={"required": ["region"]},
        )

        skill = registry.get("test")
        result = skill.validate_params({"region": "HK", "extra": "value"})

        assert result is True


class TestSkillsRegistryManagement:
    """注册表管理测试"""

    @pytest.fixture
    def registry(self):
        """创建技能注册表"""
        return MockSkillsRegistry()

    @pytest.mark.unit
    def test_clear_registry(self, registry):
        """测试清空注册表"""
        registry.register("skill1", sync_handler, "技能1")
        registry.register("skill2", sync_handler, "技能2")

        registry.clear()

        assert registry.get_skill_count() == 0
        assert registry.get_categories() == []
        assert registry.get_tags() == []

    @pytest.mark.unit
    def test_get_skill_count(self, registry):
        """测试获取技能数量"""
        assert registry.get_skill_count() == 0

        registry.register("skill1", sync_handler, "技能1")
        assert registry.get_skill_count() == 1

        registry.register("skill2", sync_handler, "技能2")
        assert registry.get_skill_count() == 2

        registry.unregister("skill1")
        assert registry.get_skill_count() == 1


class TestSkillToDict:
    """技能序列化测试"""

    @pytest.mark.unit
    def test_skill_to_dict(self):
        """测试技能转换为字典"""
        registry = MockSkillsRegistry()
        registry.register(
            "test_skill",
            sync_handler,
            description="测试技能描述",
            parameters={"required": ["id"]},
            category="test",
            tags=["unit", "test"],
        )

        skill_dict = registry.get("test_skill").to_dict()

        assert skill_dict["name"] == "test_skill"
        assert skill_dict["description"] == "测试技能描述"
        assert skill_dict["category"] == "test"
        assert "unit" in skill_dict["tags"]


class TestSkillsRegistryEdgeCases:
    """边界条件测试"""

    @pytest.fixture
    def registry(self):
        """创建技能注册表"""
        return MockSkillsRegistry()

    @pytest.mark.unit
    def test_register_with_empty_name(self, registry):
        """测试注册空名称技能"""
        # 应该能注册，但名称为空
        skill = registry.register("", sync_handler, "空名称技能")
        assert skill.name == ""

    @pytest.mark.unit
    def test_register_with_unicode_description(self, registry):
        """测试注册 Unicode 描述"""
        registry.register(
            "unicode_skill",
            sync_handler,
            "这是一个中文描述 🎉",
        )

        skill = registry.get("unicode_skill")
        assert "中文" in skill.description
        assert "🎉" in skill.description

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_empty_params(self, registry):
        """测试空参数执行"""
        registry.register(
            "empty_params",
            sync_handler,
            parameters={"required": []},
        )

        result = await registry.execute("empty_params", {})

        assert result["success"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_complex_params(self, registry):
        """测试复杂参数执行"""
        registry.register(
            "complex_params",
            sync_handler,
            parameters={"required": []},
        )

        complex_params = {
            "nested": {"key": {"deep": "value"}},
            "array": [1, 2, 3],
            "unicode": "中文测试",
        }

        result = await registry.execute("complex_params", complex_params)

        assert result["success"] is True

    @pytest.mark.slow
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_register_many_skills(self, registry):
        """测试注册大量技能"""
        for i in range(100):
            registry.register(
                f"skill_{i}",
                sync_handler,
                f"技能 {i}",
                category=f"category_{i % 10}",
            )

        assert registry.get_skill_count() == 100
        assert len(registry.get_categories()) == 10

    @pytest.mark.slow
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_skill_execution(self, registry):
        """测试并发技能执行"""
        registry.register("concurrent", async_handler, parameters={"required": []})

        tasks = [
            registry.execute("concurrent", {"index": i})
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r["success"] for r in results)


class TestSkillHandlerTypes:
    """不同类型处理器测试"""

    @pytest.fixture
    def registry(self):
        """创建技能注册表"""
        return MockSkillsRegistry()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_handler(self, registry):
        """测试同步处理器"""
        def handler(params):
            return {"sync": True, "input": params}

        registry.register("sync_test", handler, parameters={"required": []})

        result = await registry.execute("sync_test", {"key": "value"})

        assert result["success"] is True
        assert result["result"]["sync"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_handler(self, registry):
        """测试异步处理器"""
        async def handler(params):
            await asyncio.sleep(0.001)
            return {"async": True, "input": params}

        registry.register("async_test", handler, parameters={"required": []})

        result = await registry.execute("async_test", {"key": "value"})

        assert result["success"] is True
        assert result["result"]["async"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_lambda_handler(self, registry):
        """测试 Lambda 处理器"""
        registry.register(
            "lambda_test",
            lambda p: {"lambda": True, "input": p},
            parameters={"required": []},
        )

        result = await registry.execute("lambda_test", {"key": "value"})

        assert result["success"] is True
        assert result["result"]["lambda"] is True
