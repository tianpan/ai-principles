# -*- coding: utf-8 -*-
"""
Unit tests for MCP Tool Executor module

Tests cover:
- MCPToolExecutor initialization
- register_local_handler()
- unregister_local_handler()
- execute() - success and failure scenarios
- timeout handling
- retry mechanism
- caching functionality
- ToolExecutionResult dataclass
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from app.mcp.executor import MCPToolExecutor, ToolExecutionResult


class TestToolExecutionResult:
    """Tests for ToolExecutionResult dataclass"""

    def test_result_creation_success(self):
        """Test successful result creation"""
        result = ToolExecutionResult(
            success=True,
            result={"data": "value"},
            execution_time_ms=100.5,
            source="local",
        )

        assert result.success is True
        assert result.result == {"data": "value"}
        assert result.error is None
        assert result.execution_time_ms == 100.5
        assert result.source == "local"

    def test_result_creation_failure(self):
        """Test failure result creation"""
        result = ToolExecutionResult(
            success=False,
            error="Something went wrong",
            execution_time_ms=50.0,
        )

        assert result.success is False
        assert result.result is None
        assert result.error == "Something went wrong"
        assert result.source == "local"

    def test_result_to_dict(self):
        """Test to_dict() method"""
        result = ToolExecutionResult(
            success=True,
            result={"key": "value"},
            error=None,
            execution_time_ms=123.45,
            source="cache",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["result"] == {"key": "value"}
        assert result_dict["error"] is None
        assert result_dict["execution_time_ms"] == 123.45
        assert result_dict["source"] == "cache"

    def test_result_defaults(self):
        """Test default values"""
        result = ToolExecutionResult(success=True)

        assert result.result is None
        assert result.error is None
        assert result.execution_time_ms == 0
        assert result.source == "local"


class TestMCPToolExecutorInit:
    """Tests for MCPToolExecutor initialization"""

    def test_init_defaults(self):
        """Test initialization with default values"""
        executor = MCPToolExecutor()

        assert executor.timeout == 30.0
        assert executor.max_retries == 2
        assert executor.enable_cache is False

    def test_init_custom_values(self):
        """Test initialization with custom values"""
        executor = MCPToolExecutor(
            timeout_seconds=60.0,
            max_retries=5,
            enable_cache=True,
        )

        assert executor.timeout == 60.0
        assert executor.max_retries == 5
        assert executor.enable_cache is True

    def test_init_internal_state(self):
        """Test initialization of internal state"""
        executor = MCPToolExecutor()

        assert executor._local_handlers == {}
        assert executor._cache == {}
        assert executor._stats["total_calls"] == 0
        assert executor._stats["successful_calls"] == 0
        assert executor._stats["failed_calls"] == 0
        assert executor._stats["total_time_ms"] == 0


class TestRegisterLocalHandler:
    """Tests for register_local_handler()"""

    @pytest.mark.asyncio
    async def test_register_single_handler(self, executor: MCPToolExecutor, async_handler):
        """Test registering a single handler"""
        executor.register_local_handler(
            server_name="test_server",
            tool_name="test_tool",
            handler=async_handler,
        )

        assert "test_server" in executor._local_handlers
        assert "test_tool" in executor._local_handlers["test_server"]
        assert executor._local_handlers["test_server"]["test_tool"] == async_handler

    @pytest.mark.asyncio
    async def test_register_multiple_handlers_same_server(
        self, executor: MCPToolExecutor, async_handler
    ):
        """Test registering multiple handlers for same server"""
        executor.register_local_handler("server", "tool1", async_handler)
        executor.register_local_handler("server", "tool2", async_handler)

        assert len(executor._local_handlers["server"]) == 2
        assert "tool1" in executor._local_handlers["server"]
        assert "tool2" in executor._local_handlers["server"]

    @pytest.mark.asyncio
    async def test_register_handlers_different_servers(
        self, executor: MCPToolExecutor, async_handler
    ):
        """Test registering handlers for different servers"""
        executor.register_local_handler("server1", "tool", async_handler)
        executor.register_local_handler("server2", "tool", async_handler)

        assert "server1" in executor._local_handlers
        assert "server2" in executor._local_handlers
        assert "tool" in executor._local_handlers["server1"]
        assert "tool" in executor._local_handlers["server2"]

    @pytest.mark.asyncio
    async def test_register_overwrites_handler(self, executor: MCPToolExecutor, async_handler):
        """Test that registering same handler overwrites previous"""
        async def handler1(**kwargs):
            return {"version": 1}

        async def handler2(**kwargs):
            return {"version": 2}

        executor.register_local_handler("server", "tool", handler1)
        executor.register_local_handler("server", "tool", handler2)

        assert executor._local_handlers["server"]["tool"] == handler2


class TestUnregisterLocalHandler:
    """Tests for unregister_local_handler()"""

    @pytest.mark.asyncio
    async def test_unregister_existing_handler(
        self, executor: MCPToolExecutor, async_handler
    ):
        """Test unregistering an existing handler"""
        executor.register_local_handler("server", "tool", async_handler)

        result = executor.unregister_local_handler("server", "tool")

        assert result is True
        assert "tool" not in executor._local_handlers["server"]

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_handler(self, executor: MCPToolExecutor):
        """Test unregistering a non-existent handler"""
        result = executor.unregister_local_handler("server", "tool")

        assert result is False

    @pytest.mark.asyncio
    async def test_unregister_from_nonexistent_server(self, executor: MCPToolExecutor):
        """Test unregistering from non-existent server"""
        result = executor.unregister_local_handler("nonexistent", "tool")

        assert result is False

    @pytest.mark.asyncio
    async def test_unregister_keeps_other_handlers(
        self, executor: MCPToolExecutor, async_handler
    ):
        """Test that unregistering one handler keeps others"""
        executor.register_local_handler("server", "tool1", async_handler)
        executor.register_local_handler("server", "tool2", async_handler)

        executor.unregister_local_handler("server", "tool1")

        assert "tool1" not in executor._local_handlers["server"]
        assert "tool2" in executor._local_handlers["server"]


class TestExecute:
    """Tests for execute() method"""

    @pytest.mark.asyncio
    async def test_execute_success(self, executor: MCPToolExecutor, async_handler):
        """Test successful execution"""
        executor.register_local_handler("server", "tool", async_handler)

        result = await executor.execute(
            server_name="server",
            tool_name="tool",
            arguments={"param": "value"},
        )

        assert result.success is True
        assert result.result == {"success": True, "data": {"param": "value"}}
        assert result.source == "local"
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_updates_stats(self, executor: MCPToolExecutor, async_handler):
        """Test that execution updates statistics"""
        executor.register_local_handler("server", "tool", async_handler)

        stats_before = executor.get_stats()
        assert stats_before["total_calls"] == 0

        await executor.execute("server", "tool", {})

        stats_after = executor.get_stats()
        assert stats_after["total_calls"] == 1
        assert stats_after["successful_calls"] == 1
        assert stats_after["total_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_execute_nonexistent_server(self, executor: MCPToolExecutor):
        """Test execution with non-existent server"""
        result = await executor.execute(
            server_name="nonexistent",
            tool_name="tool",
            arguments={},
        )

        assert result.success is False
        assert "No handler registered for server" in result.error

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, executor: MCPToolExecutor, async_handler):
        """Test execution with non-existent tool"""
        executor.register_local_handler("server", "existing_tool", async_handler)

        result = await executor.execute(
            server_name="server",
            tool_name="nonexistent_tool",
            arguments={},
        )

        assert result.success is False
        assert "Tool not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_arguments(self, executor: MCPToolExecutor):
        """Test execution with various arguments"""
        async def handler_with_args(**kwargs):
            return {"received": kwargs}

        executor.register_local_handler("server", "tool", handler_with_args)

        result = await executor.execute(
            server_name="server",
            tool_name="tool",
            arguments={"key1": "value1", "key2": 123, "key3": [1, 2, 3]},
        )

        assert result.success is True
        assert result.result["received"]["key1"] == "value1"
        assert result.result["received"]["key2"] == 123


class TestTimeout:
    """Tests for timeout handling"""

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self, executor: MCPToolExecutor):
        """Test that timeout triggers retry and returns error"""
        async def slow_handler(**kwargs):
            await asyncio.sleep(10)
            return {"success": True}

        executor.register_local_handler("server", "tool", slow_handler)
        executor.timeout = 0.1  # Very short timeout
        executor.max_retries = 0  # No retries for faster test

        result = await executor.execute("server", "tool", {})

        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_timeout_with_retries(self, executor: MCPToolExecutor):
        """Test timeout with retries"""
        async def slow_handler(**kwargs):
            await asyncio.sleep(10)
            return {"success": True}

        executor.register_local_handler("server", "tool", slow_handler)
        executor.timeout = 0.1
        executor.max_retries = 2

        start_time = asyncio.get_event_loop().time()
        result = await executor.execute("server", "tool", {})
        elapsed = asyncio.get_event_loop().time() - start_time

        assert result.success is False
        # Should have attempted max_retries + 1 times
        assert elapsed >= 0.3  # At least 3 timeout attempts


class TestRetry:
    """Tests for retry mechanism"""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, executor: MCPToolExecutor):
        """Test retry on timeout (timeout raises and triggers retry)"""
        call_count = 0

        async def slow_handler(**kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(10)  # Will timeout
            return {"call": call_count}

        executor.register_local_handler("server", "tool", slow_handler)
        executor.timeout = 0.1  # Very short timeout
        executor.max_retries = 2

        result = await executor.execute("server", "tool", {})

        assert result.success is False
        # Timeout triggers retry: initial + max_retries
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_retry_on_timeout_with_logging(self, executor: MCPToolExecutor):
        """Test that timeout triggers retry and error logging"""
        async def slow_handler(**kwargs):
            await asyncio.sleep(10)  # Will timeout
            return {"success": True}

        executor.register_local_handler("server", "tool", slow_handler)
        executor.timeout = 0.05  # Very short timeout
        executor.max_retries = 1  # 1 retry = 2 total attempts

        with patch("app.mcp.executor.logger") as mock_logger:
            result = await executor.execute("server", "tool", {})

            assert result.success is False
            assert "Timeout" in result.error
            # Should have logged warnings for each timeout
            assert mock_logger.warning.call_count >= 2

    @pytest.mark.asyncio
    async def test_no_retry_on_handler_exception(self, executor: MCPToolExecutor):
        """Test that handler exceptions do NOT trigger retry

        Note: The current implementation only retries on TimeoutError.
        Regular exceptions return ToolExecutionResult(success=False) immediately.
        """
        call_count = 0

        async def failing_handler(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Error on call {call_count}")

        executor.register_local_handler("server", "tool", failing_handler)
        executor.max_retries = 2

        result = await executor.execute("server", "tool", {})

        assert result.success is False
        # Handler exceptions are caught in _execute_local and returned as failure
        # They don't trigger retry in the current implementation
        assert call_count == 1  # Only called once
        assert "Error on call 1" in result.error

    @pytest.mark.asyncio
    async def test_no_retries_configured(self, executor: MCPToolExecutor):
        """Test with no retries configured"""
        call_count = 0

        async def failing_handler(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Error")

        executor.register_local_handler("server", "tool", failing_handler)
        executor.max_retries = 0

        result = await executor.execute("server", "tool", {})

        assert result.success is False
        assert call_count == 1


class TestCache:
    """Tests for caching functionality"""

    @pytest.mark.asyncio
    async def test_cache_enabled(self, executor: MCPToolExecutor):
        """Test caching when enabled"""
        call_count = 0

        async def counting_handler(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        executor.register_local_handler("server", "tool", counting_handler)

        # First call
        result1 = await executor.execute(
            "server", "tool", {"key": "value"}, use_cache=True
        )

        # Second call with same arguments
        result2 = await executor.execute(
            "server", "tool", {"key": "value"}, use_cache=True
        )

        assert result1.success is True
        assert result2.success is True
        assert result1.result["count"] == 1
        assert result2.result["count"] == 1  # Cached, not called again
        assert result2.source == "cache"

    @pytest.mark.asyncio
    async def test_cache_disabled(self, executor_no_cache: MCPToolExecutor):
        """Test that cache is not used when disabled"""
        call_count = 0

        async def counting_handler(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        executor_no_cache.register_local_handler("server", "tool", counting_handler)

        result1 = await executor_no_cache.execute(
            "server", "tool", {"key": "value"}, use_cache=True
        )
        result2 = await executor_no_cache.execute(
            "server", "tool", {"key": "value"}, use_cache=True
        )

        assert result1.result["count"] == 1
        assert result2.result["count"] == 2  # Called again, not cached

    @pytest.mark.asyncio
    async def test_cache_different_arguments(self, executor: MCPToolExecutor):
        """Test that different arguments don't use cache"""
        call_count = 0

        async def counting_handler(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"count": call_count, "args": kwargs}

        executor.register_local_handler("server", "tool", counting_handler)

        result1 = await executor.execute(
            "server", "tool", {"key": "value1"}, use_cache=True
        )
        result2 = await executor.execute(
            "server", "tool", {"key": "value2"}, use_cache=True
        )

        assert result1.result["count"] == 1
        assert result2.result["count"] == 2
        assert result2.source == "local"  # Not from cache

    @pytest.mark.asyncio
    async def test_clear_cache(self, executor: MCPToolExecutor):
        """Test clearing cache"""
        async def handler(**kwargs):
            return {"data": "value"}

        executor.register_local_handler("server", "tool", handler)

        await executor.execute("server", "tool", {"key": "value"}, use_cache=True)

        assert len(executor._cache) == 1

        executor.clear_cache()

        assert len(executor._cache) == 0


class TestStatistics:
    """Tests for statistics functionality"""

    @pytest.mark.asyncio
    async def test_get_stats_initial(self, executor: MCPToolExecutor):
        """Test initial statistics"""
        stats = executor.get_stats()

        assert stats["total_calls"] == 0
        assert stats["successful_calls"] == 0
        assert stats["failed_calls"] == 0
        assert stats["total_time_ms"] == 0
        assert stats["average_time_ms"] == 0
        assert stats["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_after_calls(self, executor: MCPToolExecutor, async_handler):
        """Test statistics after multiple calls"""
        executor.register_local_handler("server", "tool", async_handler)

        await executor.execute("server", "tool", {})
        await executor.execute("server", "tool", {})

        stats = executor.get_stats()

        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 2
        assert stats["average_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_get_stats_mixed_results(self, executor: MCPToolExecutor, async_handler):
        """Test statistics with mixed success/failure"""
        # Define failing handler inline
        async def failing_handler(**kwargs):
            raise ValueError("Test error")

        executor.register_local_handler("server", "success_tool", async_handler)
        executor.register_local_handler("server", "fail_tool", failing_handler)

        await executor.execute("server", "success_tool", {})
        await executor.execute("server", "fail_tool", {})

        stats = executor.get_stats()

        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 1

    @pytest.mark.asyncio
    async def test_reset_stats(self, executor: MCPToolExecutor, async_handler):
        """Test resetting statistics"""
        executor.register_local_handler("server", "tool", async_handler)

        await executor.execute("server", "tool", {})

        assert executor.get_stats()["total_calls"] == 1

        executor.reset_stats()

        stats = executor.get_stats()
        assert stats["total_calls"] == 0
        assert stats["successful_calls"] == 0
        assert stats["failed_calls"] == 0
        assert stats["total_time_ms"] == 0

    @pytest.mark.asyncio
    async def test_stats_cache_size(self, executor: MCPToolExecutor):
        """Test cache size in statistics"""
        async def handler(**kwargs):
            return {"data": kwargs}

        executor.register_local_handler("server", "tool", handler)

        assert executor.get_stats()["cache_size"] == 0

        await executor.execute("server", "tool", {"key": "value1"}, use_cache=True)
        assert executor.get_stats()["cache_size"] == 1

        await executor.execute("server", "tool", {"key": "value2"}, use_cache=True)
        assert executor.get_stats()["cache_size"] == 2


class TestCacheKey:
    """Tests for cache key generation"""

    def test_make_cache_key_consistent(self, executor: MCPToolExecutor):
        """Test that same arguments produce same cache key"""
        key1 = executor._make_cache_key("server", "tool", {"a": 1, "b": 2})
        key2 = executor._make_cache_key("server", "tool", {"a": 1, "b": 2})

        assert key1 == key2

    def test_make_cache_key_different_servers(self, executor: MCPToolExecutor):
        """Test that different servers produce different keys"""
        key1 = executor._make_cache_key("server1", "tool", {"a": 1})
        key2 = executor._make_cache_key("server2", "tool", {"a": 1})

        assert key1 != key2

    def test_make_cache_key_different_tools(self, executor: MCPToolExecutor):
        """Test that different tools produce different keys"""
        key1 = executor._make_cache_key("server", "tool1", {"a": 1})
        key2 = executor._make_cache_key("server", "tool2", {"a": 1})

        assert key1 != key2

    def test_make_cache_key_different_args(self, executor: MCPToolExecutor):
        """Test that different arguments produce different keys"""
        key1 = executor._make_cache_key("server", "tool", {"a": 1})
        key2 = executor._make_cache_key("server", "tool", {"a": 2})

        assert key1 != key2

    def test_make_cache_key_order_independent(self, executor: MCPToolExecutor):
        """Test that argument order doesn't affect key"""
        key1 = executor._make_cache_key("server", "tool", {"a": 1, "b": 2})
        key2 = executor._make_cache_key("server", "tool", {"b": 2, "a": 1})

        assert key1 == key2


class TestEdgeCases:
    """Edge case tests"""

    @pytest.mark.asyncio
    async def test_execute_with_no_arguments(self, executor: MCPToolExecutor):
        """Test execution with no arguments"""
        async def no_args_handler(**kwargs):
            return {"called": True}

        executor.register_local_handler("server", "tool", no_args_handler)

        result = await executor.execute("server", "tool", {})

        assert result.success is True
        assert result.result == {"called": True}

    @pytest.mark.asyncio
    async def test_execute_with_none_result(self, executor: MCPToolExecutor):
        """Test handler returning None"""
        async def none_handler(**kwargs):
            return None

        executor.register_local_handler("server", "tool", none_handler)

        result = await executor.execute("server", "tool", {})

        assert result.success is True
        assert result.result is None

    @pytest.mark.asyncio
    async def test_execute_with_complex_result(self, executor: MCPToolExecutor):
        """Test handler returning complex nested data"""
        complex_data = {
            "level1": {
                "level2": {
                    "level3": [{"a": 1}, {"b": 2}],
                },
                "list": [1, 2, 3, 4, 5],
            },
            "unicode": "\u4e2d\u6587",
        }

        async def complex_handler(**kwargs):
            return complex_data

        executor.register_local_handler("server", "tool", complex_handler)

        result = await executor.execute("server", "tool", {})

        assert result.success is True
        assert result.result == complex_data

    @pytest.mark.asyncio
    async def test_execute_updates_failed_stats_on_exception(
        self, executor: MCPToolExecutor
    ):
        """Test that failed calls update statistics"""
        async def failing_handler(**kwargs):
            raise RuntimeError("Test error")

        executor.register_local_handler("server", "tool", failing_handler)
        executor.max_retries = 0

        await executor.execute("server", "tool", {})

        stats = executor.get_stats()
        assert stats["total_calls"] == 1
        assert stats["failed_calls"] == 1
        assert stats["successful_calls"] == 0
