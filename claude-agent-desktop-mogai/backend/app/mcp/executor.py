# -*- coding: utf-8 -*-
"""
MCP Tool Executor - MCP 工具执行器

负责执行 MCP 工具调用，支持本地和远程两种模式
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# 重试延迟基数（秒）
RETRY_DELAY_BASE_SECONDS = 0.5

# 默认缓存 TTL（秒），0 表示不启用 TTL
DEFAULT_CACHE_TTL_SECONDS = 300


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
    error: Optional[str] = None
    execution_time_ms: float = 0
    source: str = "local"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
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
        ttl_seconds: 生存时间（秒）
    """
    value: Any
    created_at: float
    ttl_seconds: int

    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        if self.ttl_seconds <= 0:
            return False  # 永不过期
        return time.time() - self.created_at > self.ttl_seconds


class MCPToolExecutor:
    """
    MCP 工具执行器

    支持两种执行模式：
    1. 本地执行：直接调用注册的 Python 函数
    2. 远程执行：通过 HTTP 调用远程 MCP Server

    Features:
    - 超时控制
    - 重试机制
    - 执行统计
    - Mock 数据支持
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        enable_cache: bool = False,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        max_cache_size: int = 1000,
    ):
        """
        初始化执行器

        Args:
            timeout_seconds: 执行超时时间（秒）
            max_retries: 最大重试次数
            enable_cache: 是否启用结果缓存
            cache_ttl_seconds: 缓存 TTL（秒）
            max_cache_size: 最大缓存条目数
        """
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.enable_cache = enable_cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_size = max_cache_size

        # 本地处理器注册表：server_name -> {tool_name -> handler}
        self._local_handlers: Dict[str, Dict[str, Callable[..., Awaitable[Any]]]] = {}

        # 结果缓存（带 TTL）
        self._cache: Dict[str, CacheEntry] = {}

        # 执行统计
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_time_ms": 0,
        }

    def register_local_handler(
        self,
        server_name: str,
        tool_name: str,
        handler: Callable[..., Awaitable[Any]]
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
        logger.info(f"Registered local handler: {server_name}.{tool_name}")

    def unregister_local_handler(self, server_name: str, tool_name: str) -> bool:
        """
        注销本地处理器

        Args:
            server_name: MCP Server 名称
            tool_name: 工具名称

        Returns:
            是否成功注销
        """
        if server_name in self._local_handlers:
            if tool_name in self._local_handlers[server_name]:
                del self._local_handlers[server_name][tool_name]
                return True
        return False

    async def execute(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
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
        self._stats["total_calls"] += 1

        # 检查缓存
        cache_key = self._make_cache_key(server_name, tool_name, arguments)
        if use_cache and self.enable_cache and cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if not cache_entry.is_expired():
                return ToolExecutionResult(
                    success=True,
                    result=cache_entry.value,
                    execution_time_ms=0,
                    source="cache",
                )
            else:
                # 清理过期缓存
                del self._cache[cache_key]

        # 尝试本地执行
        result = await self._execute_with_retry(
            server_name, tool_name, arguments
        )

        # 更新统计
        execution_time = (time.time() - start_time) * 1000
        result.execution_time_ms = execution_time
        self._stats["total_time_ms"] += execution_time

        if result.success:
            self._stats["successful_calls"] += 1
            # 缓存结果（带 TTL）
            if self.enable_cache and use_cache:
                self._cleanup_cache_if_needed()
                self._cache[cache_key] = CacheEntry(
                    value=result.result,
                    created_at=time.time(),
                    ttl_seconds=self.cache_ttl_seconds,
                )
        else:
            self._stats["failed_calls"] += 1

        return result

    async def _execute_with_retry(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
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
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # 尝试本地执行
                result = await self._execute_local(
                    server_name, tool_name, arguments
                )
                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout}s"
                logger.warning(
                    f"Tool execution timeout: {server_name}.{tool_name} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )

            except Exception as e:
                last_error = str(e)
                logger.error(
                    f"Tool execution error: {server_name}.{tool_name} - {e} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )

            # 等待后重试
            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))

        return ToolExecutionResult(
            success=False,
            error=last_error or "Unknown error",
        )

    async def _execute_local(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolExecutionResult:
        """
        本地执行

        Args:
            server_name: MCP Server 名称
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            ToolExecutionResult 实例
        """
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
                timeout=self.timeout
            )
            return ToolExecutionResult(
                success=True,
                result=result,
                source="local",
            )

        except asyncio.TimeoutError:
            raise

        except Exception as e:
            return ToolExecutionResult(
                success=False,
                error=str(e),
            )

    def _make_cache_key(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
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

        采用简单策略：清理过期的和最旧的一半缓存
        """
        if len(self._cache) < self.max_cache_size:
            return

        # 先清理过期缓存
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            del self._cache[key]

        # 如果还是超限，清理最旧的一半
        if len(self._cache) >= self.max_cache_size:
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1].created_at
            )
            keys_to_remove = [k for k, _ in sorted_items[:len(sorted_items) // 2]]
            for key in keys_to_remove:
                del self._cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        avg_time = (
            self._stats["total_time_ms"] / self._stats["total_calls"]
            if self._stats["total_calls"] > 0
            else 0
        )
        return {
            **self._stats,
            "average_time_ms": avg_time,
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def reset_stats(self) -> None:
        """重置统计"""
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_time_ms": 0,
        }
