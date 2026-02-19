"""
Session Manager 测试
====================

测试会话管理功能：
- 会话创建
- 会话持久化
- 历史管理
- 会话清理
- 并发访问

测试覆盖：
- 单元测试：独立方法
- 集成测试：完整会话生命周期
- 边界测试：异常情况处理
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Session Manager 类（用于测试）
# =============================================================================


class MockSessionManager:
    """
    模拟会话管理器实现

    提供会话的创建、存储、检索和管理功能。
    使用内存存储，模拟真实的会话管理行为。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_history = self.config.get("max_history", 100)
        self.session_timeout = self.config.get("timeout", 3600)  # 默认 1 小时
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._message_history: Dict[str, List[Dict[str, Any]]] = {}

    async def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        创建新会话

        Args:
            session_id: 可选的会话 ID，不提供则自动生成
            metadata: 会话元数据

        Returns:
            创建的会话信息
        """
        import uuid

        actual_session_id = session_id or str(uuid.uuid4())

        if actual_session_id in self._sessions:
            raise ValueError(f"会话 {actual_session_id} 已存在")

        now = datetime.utcnow().isoformat()
        session = {
            "session_id": actual_session_id,
            "created_at": now,
            "updated_at": now,
            "status": "active",
            "metadata": metadata or {},
            "message_count": 0,
        }

        self._sessions[actual_session_id] = session
        self._message_history[actual_session_id] = []

        return session.copy()

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息

        Args:
            session_id: 会话 ID

        Returns:
            会话信息，不存在返回 None
        """
        session = self._sessions.get(session_id)
        if session:
            return session.copy()
        return None

    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        更新会话信息

        Args:
            session_id: 会话 ID
            updates: 要更新的字段

        Returns:
            更新后的会话信息
        """
        if session_id not in self._sessions:
            raise ValueError(f"会话 {session_id} 不存在")

        session = self._sessions[session_id]
        session.update(updates)
        session["updated_at"] = datetime.utcnow().isoformat()

        return session.copy()

    async def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话 ID

        Returns:
            是否删除成功
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            if session_id in self._message_history:
                del self._message_history[session_id]
            return True
        return False

    async def add_message(
        self,
        session_id: str,
        message: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        添加消息到会话历史

        Args:
            session_id: 会话 ID
            message: 消息内容

        Returns:
            添加的消息（包含 ID 和时间戳）
        """
        if session_id not in self._sessions:
            raise ValueError(f"会话 {session_id} 不存在")

        import uuid

        message_with_meta = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            **message,
        }

        history = self._message_history[session_id]
        history.append(message_with_meta)

        # 检查是否超过最大历史记录限制
        if len(history) > self.max_history:
            # 移除最旧的消息
            removed = len(history) - self.max_history
            self._message_history[session_id] = history[removed:]

        # 更新会话
        await self.update_session(
            session_id,
            {
                "message_count": len(self._message_history[session_id]),
                "last_message_at": message_with_meta["timestamp"],
            },
        )

        return message_with_meta

    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        获取会话历史消息

        Args:
            session_id: 会话 ID
            limit: 返回消息数量限制
            offset: 偏移量

        Returns:
            消息列表
        """
        if session_id not in self._message_history:
            return []

        history = self._message_history[session_id]

        # 应用偏移和限制
        start = min(offset, len(history))
        if limit is None:
            end = len(history)
        else:
            end = min(start + limit, len(history))

        return history[start:end]

    async def clear_history(self, session_id: str) -> bool:
        """
        清除会话历史

        Args:
            session_id: 会话 ID

        Returns:
            是否清除成功
        """
        if session_id not in self._sessions:
            return False

        self._message_history[session_id] = []
        await self.update_session(session_id, {"message_count": 0})
        return True

    async def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        列出所有会话

        Args:
            status: 过滤状态
            limit: 返回数量限制

        Returns:
            会话列表
        """
        sessions = list(self._sessions.values())

        if status:
            sessions = [s for s in sessions if s.get("status") == status]

        return sessions[:limit]

    async def cleanup_expired_sessions(self) -> int:
        """
        清理过期会话

        Returns:
            清理的会话数量
        """
        now = datetime.utcnow()
        expired = []

        for session_id, session in self._sessions.items():
            updated_at = datetime.fromisoformat(session["updated_at"])
            age = (now - updated_at).total_seconds()

            if age > self.session_timeout:
                expired.append(session_id)

        for session_id in expired:
            await self.delete_session(session_id)

        return len(expired)

    async def get_session_stats(self) -> Dict[str, Any]:
        """
        获取会话统计信息

        Returns:
            统计数据
        """
        total_messages = sum(
            len(history) for history in self._message_history.values()
        )

        return {
            "total_sessions": len(self._sessions),
            "total_messages": total_messages,
            "active_sessions": len(
                [s for s in self._sessions.values() if s.get("status") == "active"]
            ),
        }


# =============================================================================
# 测试类
# =============================================================================


class TestSessionCreation:
    """会话创建测试"""

    @pytest.fixture
    def manager(self, mock_config):
        """创建会话管理器"""
        return MockSessionManager(config=mock_config.get("session", {}))

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_session_auto_id(self, manager):
        """测试自动生成 ID 创建会话"""
        session = await manager.create_session()

        assert "session_id" in session
        assert session["status"] == "active"
        assert "created_at" in session

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_session_custom_id(self, manager):
        """测试自定义 ID 创建会话"""
        custom_id = "custom-session-123"
        session = await manager.create_session(session_id=custom_id)

        assert session["session_id"] == custom_id

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_session_with_metadata(self, manager):
        """测试带元数据创建会话"""
        metadata = {"user_id": "user-001", "device": "mobile"}
        session = await manager.create_session(metadata=metadata)

        assert session["metadata"]["user_id"] == "user-001"
        assert session["metadata"]["device"] == "mobile"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_duplicate_session_raises_error(self, manager):
        """测试创建重复会话抛出异常"""
        session_id = "duplicate-test"
        await manager.create_session(session_id=session_id)

        with pytest.raises(ValueError, match="已存在"):
            await manager.create_session(session_id=session_id)


class TestSessionRetrieval:
    """会话检索测试"""

    @pytest.fixture
    async def manager(self, mock_config):
        """创建会话管理器并添加测试数据"""
        manager = MockSessionManager(config=mock_config.get("session", {}))
        return manager

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_existing_session(self, manager):
        """测试获取存在的会话"""
        created = await manager.create_session(session_id="test-session")
        session = await manager.get_session("test-session")

        assert session is not None
        assert session["session_id"] == "test-session"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_non_existent_session(self, manager):
        """测试获取不存在的会话"""
        session = await manager.get_session("non-existent")

        assert session is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_all_sessions(self, manager):
        """测试列出所有会话"""
        # 创建多个会话
        for i in range(5):
            await manager.create_session(session_id=f"session-{i}")

        sessions = await manager.list_sessions()

        assert len(sessions) == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_sessions_by_status(self, manager):
        """测试按状态过滤会话"""
        # 创建多个会话
        await manager.create_session(session_id="active-1")
        await manager.create_session(session_id="active-2")
        inactive_id = "inactive-1"
        await manager.create_session(session_id=inactive_id)
        await manager.update_session(inactive_id, {"status": "inactive"})

        active_sessions = await manager.list_sessions(status="active")

        assert len(active_sessions) == 2


class TestSessionUpdate:
    """会话更新测试"""

    @pytest.fixture
    async def manager(self, mock_config):
        """创建会话管理器"""
        manager = MockSessionManager(config=mock_config.get("session", {}))
        await manager.create_session(session_id="test-session")
        return manager

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_session_fields(self, manager):
        """测试更新会话字段"""
        updated = await manager.update_session(
            "test-session",
            {"status": "paused", "custom_field": "value"},
        )

        assert updated["status"] == "paused"
        assert updated["custom_field"] == "value"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_updates_timestamp(self, manager):
        """测试更新自动更新时间戳"""
        import asyncio

        original = await manager.get_session("test-session")
        await asyncio.sleep(0.01)  # 确保时间差异

        updated = await manager.update_session("test-session", {"key": "value"})

        assert updated["updated_at"] != original["updated_at"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_non_existent_session_raises_error(self, manager):
        """测试更新不存在的会话抛出异常"""
        with pytest.raises(ValueError, match="不存在"):
            await manager.update_session("non-existent", {"key": "value"})


class TestSessionDeletion:
    """会话删除测试"""

    @pytest.fixture
    async def manager(self, mock_config):
        """创建会话管理器"""
        manager = MockSessionManager(config=mock_config.get("session", {}))
        await manager.create_session(session_id="to-delete")
        return manager

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_existing_session(self, manager):
        """测试删除存在的会话"""
        result = await manager.delete_session("to-delete")

        assert result is True
        session = await manager.get_session("to-delete")
        assert session is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_non_existent_session(self, manager):
        """测试删除不存在的会话"""
        result = await manager.delete_session("non-existent")

        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_clears_message_history(self, manager):
        """测试删除会话清除消息历史"""
        # 添加消息
        await manager.add_message("to-delete", {"role": "user", "content": "测试"})

        await manager.delete_session("to-delete")

        # 历史应该被清除
        history = await manager.get_history("to-delete")
        assert history == []


class TestMessageHistory:
    """消息历史测试"""

    @pytest.fixture
    async def manager(self, mock_config):
        """创建会话管理器"""
        manager = MockSessionManager(config=mock_config.get("session", {}))
        await manager.create_session(session_id="history-test")
        return manager

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_message(self, manager):
        """测试添加消息"""
        message = await manager.add_message(
            "history-test",
            {"role": "user", "content": "你好"},
        )

        assert "message_id" in message
        assert "timestamp" in message
        assert message["role"] == "user"
        assert message["content"] == "你好"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_history(self, manager):
        """测试获取历史"""
        # 添加多条消息
        for i in range(5):
            await manager.add_message(
                "history-test",
                {"role": "user", "content": f"消息 {i}"},
            )

        history = await manager.get_history("history-test")

        assert len(history) == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_history_with_limit(self, manager):
        """测试限制历史数量"""
        for i in range(10):
            await manager.add_message(
                "history-test",
                {"role": "user", "content": f"消息 {i}"},
            )

        history = await manager.get_history("history-test", limit=5)

        assert len(history) == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_history_with_offset(self, manager):
        """测试历史偏移"""
        for i in range(10):
            await manager.add_message(
                "history-test",
                {"role": "user", "content": f"消息 {i}"},
            )

        history = await manager.get_history("history-test", offset=5, limit=3)

        assert len(history) == 3
        # 应该从索引 5 开始
        assert "消息 5" in history[0]["content"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_history(self, manager):
        """测试清除历史"""
        await manager.add_message(
            "history-test",
            {"role": "user", "content": "测试消息"},
        )

        result = await manager.clear_history("history-test")

        assert result is True
        history = await manager.get_history("history-test")
        assert len(history) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_history_max_limit(self):
        """测试历史最大限制"""
        config = {"max_history": 5}
        manager = MockSessionManager(config=config)
        await manager.create_session(session_id="limit-test")

        # 添加超过限制的消息
        for i in range(10):
            await manager.add_message(
                "limit-test",
                {"role": "user", "content": f"消息 {i}"},
            )

        history = await manager.get_history("limit-test")

        # 应该只保留最新的 5 条
        assert len(history) == 5
        # 最新的消息应该在
        assert "消息 9" in history[-1]["content"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_message_to_non_existent_session_raises_error(self, manager):
        """测试向不存在的会话添加消息抛出异常"""
        with pytest.raises(ValueError, match="不存在"):
            await manager.add_message(
                "non-existent",
                {"role": "user", "content": "测试"},
            )


class TestSessionPersistence:
    """会话持久化测试"""

    @pytest.fixture
    def manager(self, mock_config):
        """创建会话管理器"""
        return MockSessionManager(config=mock_config.get("session", {}))

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_data_survives_operations(self, manager):
        """测试会话数据在操作后保持"""
        # 创建会话
        session = await manager.create_session(
            session_id="persist-test",
            metadata={"user_id": "user-001", "preferences": {"language": "zh"}},
        )

        # 添加消息
        await manager.add_message(
            "persist-test",
            {"role": "user", "content": "消息 1"},
        )
        await manager.add_message(
            "persist-test",
            {"role": "assistant", "content": "回复 1"},
        )

        # 获取会话
        retrieved = await manager.get_session("persist-test")

        assert retrieved["metadata"]["user_id"] == "user-001"
        assert retrieved["message_count"] == 2

        # 获取历史
        history = await manager.get_history("persist-test")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


class TestSessionCleanup:
    """会话清理测试"""

    @pytest.fixture
    def manager(self):
        """创建短超时的会话管理器"""
        return MockSessionManager(config={"timeout": 1})  # 1 秒超时

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, manager):
        """测试清理过期会话"""
        # 创建会话
        await manager.create_session(session_id="old-session")

        # 等待超时
        await asyncio.sleep(1.1)

        # 创建新会话
        await manager.create_session(session_id="new-session")

        # 清理过期会话
        cleaned = await manager.cleanup_expired_sessions()

        assert cleaned == 1
        assert await manager.get_session("old-session") is None
        assert await manager.get_session("new-session") is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_stats(self, manager):
        """测试获取会话统计"""
        # 创建多个会话并添加消息
        for i in range(3):
            await manager.create_session(session_id=f"stats-{i}")
            for j in range(5):
                await manager.add_message(
                    f"stats-{i}",
                    {"role": "user", "content": f"消息 {j}"},
                )

        stats = await manager.get_session_stats()

        assert stats["total_sessions"] == 3
        assert stats["total_messages"] == 15
        assert stats["active_sessions"] == 3


class TestConcurrentAccess:
    """并发访问测试"""

    @pytest.fixture
    def manager(self, mock_config):
        """创建会话管理器"""
        return MockSessionManager(config=mock_config.get("session", {}))

    @pytest.mark.slow
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self, manager):
        """测试并发创建会话"""
        tasks = [
            manager.create_session(session_id=f"concurrent-{i}")
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result["session_id"] == f"concurrent-{i}"

    @pytest.mark.slow
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_message_adding(self, manager):
        """测试并发添加消息"""
        await manager.create_session(session_id="concurrent-msg")

        tasks = [
            manager.add_message(
                "concurrent-msg",
                {"role": "user", "content": f"消息 {i}"},
            )
            for i in range(50)
        ]

        await asyncio.gather(*tasks)

        history = await manager.get_history("concurrent-msg")
        assert len(history) == 50


class TestSessionEdgeCases:
    """边界条件测试"""

    @pytest.fixture
    def manager(self, mock_config):
        """创建会话管理器"""
        return MockSessionManager(config=mock_config.get("session", {}))

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_session_id_handling(self, manager):
        """测试空会话 ID 处理"""
        # 自动生成 ID 应该能处理
        session = await manager.create_session(session_id=None)
        assert session["session_id"] is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_metadata(self, manager):
        """测试大量元数据"""
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        session = await manager.create_session(
            session_id="large-meta",
            metadata=large_metadata,
        )

        assert len(session["metadata"]) == 100

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, manager):
        """测试内容中的特殊字符"""
        await manager.create_session(session_id="special-chars")

        special_content = "!@#$%^&*(){}[]|\\:;\"'<>,.?/~`\n\t\r中文日文한국어"
        message = await manager.add_message(
            "special-chars",
            {"role": "user", "content": special_content},
        )

        assert message["content"] == special_content

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unicode_metadata(self, manager):
        """测试 Unicode 元数据"""
        unicode_metadata = {
            "用户名": "张三",
            "语言": "中文",
            "备注": "这是一个测试🎉",
        }

        session = await manager.create_session(
            session_id="unicode-meta",
            metadata=unicode_metadata,
        )

        assert session["metadata"]["用户名"] == "张三"
        assert session["metadata"]["备注"] == "这是一个测试🎉"
