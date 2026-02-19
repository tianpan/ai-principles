"""
Agent Engine 测试
==================

测试 Agent Engine 核心功能：
- 消息处理
- 流式输出
- 技能调用
- 上下文管理
- 错误处理

测试策略：
1. 单元测试 - 测试独立方法
2. 集成测试 - 测试消息处理流程
3. 边界测试 - 测试异常输入
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Agent Engine 类（用于测试）
# =============================================================================


class MockAgentEngine:
    """
    模拟 Agent Engine 实现

    用于测试的轻量级实现，模拟真实 Agent Engine 的行为。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = self.config.get("model", {})
        self.skills_registry = {}
        self.context_manager = {}
        self._is_initialized = False

    async def initialize(self) -> None:
        """初始化 Agent Engine"""
        await asyncio.sleep(0.01)  # 模拟初始化延迟
        self._is_initialized = True

    async def process_message(
        self,
        message: str,
        session_id: str = None,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        处理用户消息

        Args:
            message: 用户输入消息
            session_id: 会话 ID
            context: 上下文信息

        Returns:
            处理结果，包含响应内容
        """
        if not self._is_initialized:
            raise RuntimeError("Agent Engine 未初始化")

        if not message or not message.strip():
            raise ValueError("消息不能为空")

        # 模拟处理延迟
        await asyncio.sleep(0.01)

        # 检查是否需要调用技能
        if "查询价格" in message or "价格" in message:
            skill_result = await self._invoke_skill("gas_price_query", {})
            return {
                "response": f"查询到的价格信息：{skill_result}",
                "skill_called": "gas_price_query",
                "session_id": session_id,
            }

        # 普通响应
        return {
            "response": f"收到您的消息：{message}",
            "session_id": session_id,
            "model": self.model.get("name", "unknown"),
        }

    async def stream_response(
        self,
        message: str,
        session_id: str = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式输出响应

        Args:
            message: 用户输入消息
            session_id: 会话 ID

        Yields:
            响应文本片段
        """
        if not self._is_initialized:
            raise RuntimeError("Agent Engine 未初始化")

        # 模拟流式输出
        response = f"收到您的消息：{message}"
        for char in response:
            await asyncio.sleep(0.001)
            yield char

    async def invoke_skill(
        self,
        skill_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        调用技能

        Args:
            skill_name: 技能名称
            params: 技能参数

        Returns:
            技能执行结果
        """
        return await self._invoke_skill(skill_name, params)

    async def _invoke_skill(
        self,
        skill_name: str,
        params: Dict[str, Any],
    ) -> Any:
        """内部技能调用实现"""
        if skill_name not in self.skills_registry:
            # 模拟技能执行
            return {"result": f"执行技能 {skill_name} 成功", "params": params}
        return await self.skills_registry[skill_name].execute(params)

    def register_skill(self, skill_name: str, handler: Any) -> None:
        """注册技能"""
        self.skills_registry[skill_name] = handler

    async def get_context(self, session_id: str) -> Dict[str, Any]:
        """获取会话上下文"""
        return self.context_manager.get(session_id, {})

    async def update_context(
        self,
        session_id: str,
        context: Dict[str, Any],
    ) -> None:
        """更新会话上下文"""
        if session_id not in self.context_manager:
            self.context_manager[session_id] = {}
        self.context_manager[session_id].update(context)


# =============================================================================
# 测试类
# =============================================================================


class TestAgentEngineInitialization:
    """Agent Engine 初始化测试"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config):
        """测试成功初始化"""
        engine = MockAgentEngine(config=mock_config)
        assert not engine._is_initialized

        await engine.initialize()

        assert engine._is_initialized

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_with_empty_config(self):
        """测试空配置初始化"""
        engine = MockAgentEngine(config={})

        await engine.initialize()

        assert engine._is_initialized
        assert engine.model == {}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_with_none_config(self):
        """测试 None 配置初始化"""
        engine = MockAgentEngine(config=None)

        await engine.initialize()

        assert engine._is_initialized


class TestMessageProcessing:
    """消息处理测试"""

    @pytest.fixture
    async def engine(self, mock_config):
        """创建并初始化的 Agent Engine"""
        engine = MockAgentEngine(config=mock_config)
        await engine.initialize()
        return engine

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_simple_message(self, engine):
        """测试处理简单消息"""
        result = await engine.process_message("你好")

        assert "response" in result
        assert "你好" in result["response"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_with_session(self, engine):
        """测试带会话 ID 的消息处理"""
        session_id = "test-session-123"
        result = await engine.process_message(
            message="你好",
            session_id=session_id,
        )

        assert result["session_id"] == session_id

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_with_context(self, engine):
        """测试带上下文的消息处理"""
        context = {"user_id": "user-001", "language": "zh-CN"}
        result = await engine.process_message(
            message="你好",
            context=context,
        )

        assert "response" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_empty_message_raises_error(self, engine):
        """测试空消息抛出异常"""
        with pytest.raises(ValueError, match="消息不能为空"):
            await engine.process_message("")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_whitespace_message_raises_error(self, engine):
        """测试纯空白消息抛出异常"""
        with pytest.raises(ValueError, match="消息不能为空"):
            await engine.process_message("   \n\t  ")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_before_init_raises_error(self, mock_config):
        """测试未初始化时处理消息抛出异常"""
        engine = MockAgentEngine(config=mock_config)
        # 不调用 initialize()

        with pytest.raises(RuntimeError, match="未初始化"):
            await engine.process_message("你好")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_long_message(self, engine):
        """测试处理长消息"""
        long_message = "这是一段很长的消息。" * 1000
        result = await engine.process_message(long_message)

        assert "response" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_message_with_special_characters(self, engine):
        """测试处理特殊字符消息"""
        special_chars = "!@#$%^&*(){}[]|\\:;\"'<>,.?/~`中文日文한국어"
        result = await engine.process_message(special_chars)

        assert "response" in result


class TestStreamingOutput:
    """流式输出测试"""

    @pytest.fixture
    async def engine(self, mock_config):
        """创建并初始化的 Agent Engine"""
        engine = MockAgentEngine(config=mock_config)
        await engine.initialize()
        return engine

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_response_yields_characters(self, engine):
        """测试流式输出字符"""
        chunks = []
        async for chunk in engine.stream_response("测试"):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert "测试" in full_response

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_response_is_async_generator(self, engine):
        """测试流式输出是异步生成器"""
        result = engine.stream_response("测试")

        assert hasattr(result, "__aiter__")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_response_before_init_raises_error(self, mock_config):
        """测试未初始化时流式输出抛出异常"""
        engine = MockAgentEngine(config=mock_config)

        with pytest.raises(RuntimeError, match="未初始化"):
            async for _ in engine.stream_response("测试"):
                pass


class TestSkillInvocation:
    """技能调用测试"""

    @pytest.fixture
    async def engine(self, mock_config):
        """创建并初始化的 Agent Engine"""
        engine = MockAgentEngine(config=mock_config)
        await engine.initialize()
        return engine

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invoke_skill_success(self, engine):
        """测试成功调用技能"""
        result = await engine.invoke_skill(
            skill_name="gas_price_query",
            params={"region": "香港"},
        )

        assert "result" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invoke_skill_with_empty_params(self, engine):
        """测试空参数调用技能"""
        result = await engine.invoke_skill(
            skill_name="test_skill",
            params={},
        )

        assert "result" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_register_and_invoke_custom_skill(self, engine):
        """测试注册并调用自定义技能"""
        # 创建模拟技能处理器
        mock_handler = AsyncMock()
        mock_handler.execute = AsyncMock(return_value={"custom": "result"})
        engine.register_skill("custom_skill", mock_handler)

        result = await engine.invoke_skill(
            skill_name="custom_skill",
            params={"key": "value"},
        )

        assert result == {"custom": "result"}
        mock_handler.execute.assert_called_once_with({"key": "value"})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_skill_triggered_by_message(self, engine):
        """测试消息触发技能调用"""
        result = await engine.process_message("查询价格")

        assert result.get("skill_called") == "gas_price_query"


class TestContextManagement:
    """上下文管理测试"""

    @pytest.fixture
    async def engine(self, mock_config):
        """创建并初始化的 Agent Engine"""
        engine = MockAgentEngine(config=mock_config)
        await engine.initialize()
        return engine

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_context(self, engine):
        """测试更新上下文"""
        session_id = "session-001"
        context = {"user_name": "张三", "preferences": {"language": "zh"}}

        await engine.update_context(session_id, context)

        stored = await engine.get_context(session_id)
        assert stored["user_name"] == "张三"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_empty_context(self, engine):
        """测试获取不存在的上下文"""
        context = await engine.get_context("non-existent-session")

        assert context == {}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_context_merges(self, engine):
        """测试上下文合并"""
        session_id = "session-002"

        # 第一次更新
        await engine.update_context(session_id, {"key1": "value1"})
        # 第二次更新
        await engine.update_context(session_id, {"key2": "value2"})

        context = await engine.get_context(session_id)
        assert context["key1"] == "value1"
        assert context["key2"] == "value2"


class TestAgentEngineEdgeCases:
    """边界条件测试"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self, mock_config):
        """测试并发消息处理"""
        engine = MockAgentEngine(config=mock_config)
        await engine.initialize()

        # 并发发送多个消息
        tasks = [
            engine.process_message(f"消息 {i}", session_id=f"session-{i}")
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result["session_id"] == f"session-{i}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unicode_message_handling(self, mock_config):
        """测试 Unicode 消息处理"""
        engine = MockAgentEngine(config=mock_config)
        await engine.initialize()

        unicode_messages = [
            "你好世界",
            "🎉🎊🎈",  # Emoji
            "日本語テスト",
            "한국어 테스트",
            "Привет мир",  # 俄语
        ]

        for msg in unicode_messages:
            result = await engine.process_message(msg)
            assert "response" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sql_injection_attempt(self, mock_config):
        """测试 SQL 注入尝试"""
        engine = MockAgentEngine(config=mock_config)
        await engine.initialize()

        # 模拟 SQL 注入尝试
        malicious_input = "'; DROP TABLE users; --"
        result = await engine.process_message(malicious_input)

        # 应该正常处理，不抛出异常
        assert "response" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_xss_attempt(self, mock_config):
        """测试 XSS 尝试"""
        engine = MockAgentEngine(config=mock_config)
        await engine.initialize()

        # 模拟 XSS 尝试
        malicious_input = "<script>alert('xss')</script>"
        result = await engine.process_message(malicious_input)

        # 应该正常处理
        assert "response" in result


class TestAgentEnginePerformance:
    """性能测试"""

    @pytest.mark.slow
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_context_handling(self, mock_config):
        """测试大量上下文处理"""
        engine = MockAgentEngine(config=mock_config)
        await engine.initialize()

        # 创建大量上下文数据
        large_context = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}

        await engine.update_context("large-session", large_context)

        context = await engine.get_context("large-session")
        assert len(context) == 1000

    @pytest.mark.slow
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rapid_sequential_messages(self, mock_config):
        """测试快速连续消息"""
        engine = MockAgentEngine(config=mock_config)
        await engine.initialize()

        # 快速发送 100 条消息
        for i in range(100):
            result = await engine.process_message(f"消息 {i}")
            assert "response" in result
