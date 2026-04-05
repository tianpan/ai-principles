# -*- coding: utf-8 -*-
"""
MCP Tool Executor - MCP 工具执行器

负责执行 MCP 工具调用，支持本地和远程两种模式

Features:
    - 超时控制
    - 重试机制
    - 结果缓存（带 TTL）
    - 执行统计
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


# MCP 异常类（本地定义以避免循环导入）
class MCPHandlerNotRegisteredError(Exception):
    """MCP 处理器未注册异常"""

    def __init__(self, server_name: str, tool_name: str) -> None:
        self.server_name = server_name
        self.tool_name = tool_name
        super().__init__(f"No handler registered for tool: {server_name}.{tool_name}")


class MCPTimeoutError(Exception):
    """MCP 执行超时异常"""

    def __init__(self, server_name: str, tool_name: str, timeout_seconds: float) -> None:
        self.server_name = server_name
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Tool execution timed out after {timeout_seconds}s: {server_name}.{tool_name}"
        )


# 配置模块级日志
logger = logging.getLogger(__name__)

# 常量定义
RETRY_DELAY_BASE_SECONDS: float = 0.5
DEFAULT_CACHE_TTL_SECONDS: int = 300
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_RETRIES: int = 2
DEFAULT_MAX_CACHE_SIZE: int = 1000


@dataclass
class ToolExecutionResult:
    """
    工具执行结果

    Attributes:
        success: 是否成功
        result: 执行结果（成功时）
        error: 错误信息（失败时）
        execution_time_ms: 执行耗时（毫秒）
        source: 结果来源（"local", "cache", "mock"）
    """

    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    source: str = "local"

    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典格式

        Returns:
            包含所有属性的字典
        """
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "source": self.source,
        }


@dataclass
class CacheEntry:
    """
    缓存条目

    Attributes:
        value: 缓存的值
        created_at: 创建时间戳
        ttl_seconds: 生存时间（秒），0 表示永不过期
    """

    value: Any
    created_at: float
    ttl_seconds: int

    def is_expired(self) -> bool:
        """
        检查缓存是否过期

        Returns:
            如果缓存已过期返回 True，否则返回 False
        """
        if self.ttl_seconds <= 0:
            return False
        return time.time() - self.created_at > self.ttl_seconds


@dataclass
class ExecutionStats:
    """执行统计数据"""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_time_ms: float = 0.0

    def record_success(self, execution_time_ms: float) -> None:
        """记录成功执行"""
        self.total_calls += 1
        self.successful_calls += 1
        self.total_time_ms += execution_time_ms

    def record_failure(self, execution_time_ms: float) -> None:
        """记录失败执行"""
        self.total_calls += 1
        self.failed_calls += 1
        self.total_time_ms += execution_time_ms

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        avg_time = (
            self.total_time_ms / self.total_calls if self.total_calls > 0 else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_time_ms": self.total_time_ms,
            "average_time_ms": avg_time,
        }

    def reset(self) -> None:
        """重置统计"""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_time_ms = 0.0


class MCPToolExecutor:
    """
    MCP 工具执行器

    支持两种执行模式：
        1. 本地执行：直接调用注册的 Python 函数
        2. 远程执行：通过 HTTP 调用远程 MCP Server（预留）

    Example:
        >>> executor = MCPToolExecutor(timeout_seconds=30.0)
        >>> executor.register_local_handler("top", "query_station", handler_func)
        >>> result = await executor.execute("top", "query_station", {"id": "123"})
    """

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        enable_cache: bool = False,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
    ) -> None:
        """
        初始化执行器

        Args:
            timeout_seconds: 执行超时时间（秒）
            max_retries: 最大重试次数
            enable_cache: 是否启用结果缓存
            cache_ttl_seconds: 缓存 TTL（秒）
            max_cache_size: 最大缓存条目数
        """
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._enable_cache = enable_cache
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_cache_size = max_cache_size

        # 本地处理器注册表: server_name -> {tool_name -> handler}
        self._local_handlers: dict[str, dict[str, Callable[..., Awaitable[Any]]]] = {}

        # 结果缓存（带 TTL）
        self._cache: dict[str, CacheEntry] = {}

        # 执行统计
        self._stats = ExecutionStats()

        logger.debug(
            "MCPToolExecutor initialized: timeout=%.1fs, max_retries=%d, "
            "enable_cache=%s, cache_ttl=%ds",
            timeout_seconds,
            max_retries,
            enable_cache,
            cache_ttl_seconds,
        )

    @property
    def timeout(self) -> float:
        """获取超时时间（秒）"""
        return self._timeout

    @property
    def max_retries(self) -> int:
        """获取最大重试次数"""
        return self._max_retries

    @property
    def enable_cache(self) -> bool:
        """获取是否启用缓存"""
        return self._enable_cache

    @property
    def cache_ttl_seconds(self) -> int:
        """获取缓存 TTL（秒）"""
        return self._cache_ttl_seconds

    @property
    def max_cache_size(self) -> int:
        """获取最大缓存大小"""
        return self._max_cache_size

    def register_local_handler(
        self,
        server_name: str,
        tool_name: str,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """
        注册本地处理器

        Args:
            server_name: MCP Server 名称
            tool_name: 工具名称
            handler: 处理函数（异步）
        """
        if server_name not in self._local_handlers:
            self._local_handlers[server_name] = {}

        self._local_handlers[server_name][tool_name] = handler
        logger.info(
            "Registered local handler: server=%s, tool=%s", server_name, tool_name
        )

    def unregister_local_handler(self, server_name: str, tool_name: str) -> bool:
        """
        注销本地处理器

        Args:
            server_name: MCP Server 名称
            tool_name: 工具名称

        Returns:
            如果成功注销返回 True，否则返回 False
        """
        if server_name in self._local_handlers:
            if tool_name in self._local_handlers[server_name]:
                del self._local_handlers[server_name][tool_name]
                logger.info(
                    "Unregistered local handler: server=%s, tool=%s",
                    server_name,
                    tool_name,
                )
                return True
        return False

    async def execute(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        use_cache: bool = False,
    ) -> ToolExecutionResult:
        """
        执行 MCP 工具

        Args:
            server_name: MCP Server 名称
            tool_name: 工具名称
            arguments: 工具参数
            use_cache: 是否使用缓存

        Returns:
            ToolExecutionResult 实例
        """
        start_time = time.time()

        # 生成缓存键
        cache_key = self._make_cache_key(server_name, tool_name, arguments)

        # 检查缓存
        if use_cache and self._enable_cache:
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                logger.debug(
                    "Cache hit for tool: server=%s, tool=%s", server_name, tool_name
                )
                return cached_result

        # 执行工具（带重试）
        result = await self._execute_with_retry(server_name, tool_name, arguments)

        # 计算执行时间
        execution_time_ms = (time.time() - start_time) * 1000
        result.execution_time_ms = execution_time_ms

        # 更新统计
        if result.success:
            self._stats.record_success(execution_time_ms)
            # 缓存成功结果
            if self._enable_cache and use_cache:
                self._cache_result(cache_key, result.result)
        else:
            self._stats.record_failure(execution_time_ms)

        return result

    def _get_cached_result(self, cache_key: str) -> ToolExecutionResult | None:
        """
        获取缓存的结果

        Args:
            cache_key: 缓存键

        Returns:
            如果缓存命中且未过期返回 ToolExecutionResult，否则返回 None
        """
        if cache_key not in self._cache:
            return None

        cache_entry = self._cache[cache_key]
        if cache_entry.is_expired():
            del self._cache[cache_key]
            return None

        return ToolExecutionResult(
            success=True,
            result=cache_entry.value,
            execution_time_ms=0.0,
            source="cache",
        )

    def _cache_result(self, cache_key: str, value: Any) -> None:
        """
        缓存执行结果

        Args:
            cache_key: 缓存键
            value: 要缓存的值
        """
        self._cleanup_cache_if_needed()
        self._cache[cache_key] = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl_seconds=self._cache_ttl_seconds,
        )

    async def _execute_with_retry(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        """
        带重试的执行

        Args:
            server_name: MCP Server 名称
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            ToolExecutionResult 实例
        """
        last_error: str | None = None
        total_attempts = self._max_retries + 1

        for attempt in range(total_attempts):
            try:
                result = await self._execute_local(server_name, tool_name, arguments)
                return result

            except MCPTimeoutError as e:
                last_error = str(e)
                logger.warning(
                    "Tool execution timeout: server=%s, tool=%s, attempt=%d/%d",
                    server_name,
                    tool_name,
                    attempt + 1,
                    total_attempts,
                )

            except MCPHandlerNotRegisteredError as e:
                # 处理器未注册，不需要重试
                return ToolExecutionResult(success=False, error=str(e))

            except Exception as e:
                last_error = str(e)
                logger.error(
                    "Tool execution error: server=%s, tool=%s, error=%s, attempt=%d/%d",
                    server_name,
                    tool_name,
                    e,
                    attempt + 1,
                    total_attempts,
                    exc_info=True,
                )

            # 等待后重试（指数退避）
            if attempt < self._max_retries:
                delay = RETRY_DELAY_BASE_SECONDS * (attempt + 1)
                logger.debug("Retrying in %.1f seconds...", delay)
                await self._async_sleep(delay)

        return ToolExecutionResult(
            success=False,
            error=last_error or "Unknown error",
        )

    async def _async_sleep(self, seconds: float) -> None:
        """异步睡眠（用于测试时 mock）"""
        import asyncio

        await asyncio.sleep(seconds)

    async def _execute_local(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        """
        本地执行

        Args:
            server_name: MCP Server 名称
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            ToolExecutionResult 实例

        Raises:
            MCPHandlerNotRegisteredError: 处理器未注册
            MCPTimeoutError: 执行超时
        """
        import asyncio

        # 查找本地处理器
        if server_name not in self._local_handlers:
            return ToolExecutionResult(
                success=False,
                error=f"No handler registered for server: {server_name}",
            )

        handlers = self._local_handlers[server_name]
        if tool_name not in handlers:
            return ToolExecutionResult(
                success=False,
                error=f"Tool not found: {tool_name} in server {server_name}",
            )

        handler = handlers[tool_name]

        # 执行处理器（带超时）
        try:
            result = await asyncio.wait_for(
                handler(**arguments),
                timeout=self._timeout,
            )
            return ToolExecutionResult(
                success=True,
                result=result,
                source="local",
            )

        except asyncio.TimeoutError as e:
            raise MCPTimeoutError(server_name, tool_name, self._timeout) from e

        except Exception as e:
            logger.error(
                "Local execution failed: server=%s, tool=%s, error=%s",
                server_name,
                tool_name,
                e,
                exc_info=True,
            )
            return ToolExecutionResult(
                success=False,
                error=str(e),
            )

    def _make_cache_key(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        生成稳定的缓存键

        使用 SHA256 哈希确保键的唯一性和稳定性

        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            缓存键字符串
        """
        args_str = json.dumps(arguments, sort_keys=True, default=str)
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]
        return f"{server_name}:{tool_name}:{args_hash}"

    def _cleanup_cache_if_needed(self) -> None:
        """
        清理缓存（如果超过最大大小）

        采用策略：
        1. 先清理过期的缓存
        2. 如果还是超限，清理最旧的一半缓存
        """
        if len(self._cache) < self._max_cache_size:
            return

        # 清理过期缓存
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug("Cleaned up %d expired cache entries", len(expired_keys))

        # 如果还是超限，清理最旧的一半
        if len(self._cache) >= self._max_cache_size:
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1].created_at,
            )
            keys_to_remove = [k for k, _ in sorted_items[: len(sorted_items) // 2]]
            for key in keys_to_remove:
                del self._cache[key]

            logger.debug(
                "Cleaned up %d oldest cache entries (cache was full)",
                len(keys_to_remove),
            )

    def get_stats(self) -> dict[str, Any]:
        """
        获取执行统计

        Returns:
            包含统计信息的字典
        """
        stats = self._stats.to_dict()
        stats["cache_size"] = len(self._cache)
        return stats

    def clear_cache(self) -> None:
        """清空缓存"""
        count = len(self._cache)
        self._cache.clear()
        logger.info("Cache cleared: %d entries removed", count)

    def reset_stats(self) -> None:
        """重置统计"""
        self._stats.reset()
        logger.info("Execution stats reset")
