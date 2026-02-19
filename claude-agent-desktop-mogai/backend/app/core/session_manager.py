# -*- coding: utf-8 -*-
"""
Session Manager - 会话管理模块

负责管理用户会话，包括：
- 会话创建、获取、删除
- 会话持久化（JSON 文件）
- 历史消息管理
- 会话过期清理
"""

import os
import json
import uuid
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import aiofiles

from .config import settings
from .agent_engine import Message


@dataclass
class Session:
    """
    会话数据类

    Attributes:
        id: 会话唯一标识
        title: 会话标题
        created_at: 创建时间
        updated_at: 更新时间
        messages: 消息列表
        metadata: 额外元数据
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = field(default="新会话")
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """
        添加消息到会话

        Args:
            message: 要添加的消息
        """
        self.messages.append(message.to_dict())
        self.updated_at = datetime.now().isoformat()

        # 如果是第一条用户消息，用其内容作为标题
        if len(self.messages) == 1 and message.role == "user":
            self.title = message.content[:50] + ("..." if len(message.content) > 50 else "")

    def get_messages(self) -> List[Message]:
        """
        获取消息列表

        Returns:
            Message 对象列表
        """
        return [Message.from_dict(msg) for msg in self.messages]

    def clear_messages(self) -> None:
        """清空消息历史"""
        self.messages = []
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            包含所有属性的字典
        """
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """
        从字典创建会话实例

        Args:
            data: 会话数据字典

        Returns:
            Session 实例
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", "新会话"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            messages=data.get("messages", []),
            metadata=data.get("metadata", {})
        )


class SessionManager:
    """
    会话管理器

    负责会话的生命周期管理，包括创建、获取、更新、删除和持久化

    Attributes:
        data_dir: 数据存储目录
        sessions: 内存中的会话缓存
        expire_hours: 会话过期时间（小时）
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        expire_hours: Optional[int] = None
    ):
        """
        初始化会话管理器

        Args:
            data_dir: 数据存储目录（可选，默认使用配置中的目录）
            expire_hours: 会话过期时间（可选，默认使用配置中的值）
        """
        self.data_dir = data_dir or settings.get_data_path("sessions")
        self.expire_hours = expire_hours or settings.session_expire_hours

        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)

        # 内存缓存
        self._sessions: Dict[str, Session] = {}

        # 锁，用于并发控制 - 延迟初始化
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """
        获取锁实例（延迟初始化）

        在异步上下文中创建锁，避免在非异步环境中创建的错误

        Returns:
            asyncio.Lock 实例
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_session_file(self, session_id: str) -> str:
        """
        获取会话文件路径

        Args:
            session_id: 会话 ID

        Returns:
            会话文件的完整路径
        """
        return os.path.join(self.data_dir, f"{session_id}.json")

    async def _load_session(self, session_id: str) -> Optional[Session]:
        """
        从文件加载会话

        Args:
            session_id: 会话 ID

        Returns:
            Session 实例，如果文件不存在则返回 None
        """
        file_path = self._get_session_file(session_id)

        if not os.path.exists(file_path):
            return None

        try:
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                return Session.from_dict(data)
        except (json.JSONDecodeError, IOError) as e:
            # 文件损坏或读取错误
            print(f"加载会话 {session_id} 失败: {e}")
            return None

    async def _save_session(self, session: Session) -> bool:
        """
        保存会话到文件

        Args:
            session: 要保存的会话

        Returns:
            保存成功返回 True，否则返回 False
        """
        file_path = self._get_session_file(session.id)

        try:
            async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(session.to_dict(), ensure_ascii=False, indent=2))
            return True
        except IOError as e:
            print(f"保存会话 {session.id} 失败: {e}")
            return False

    async def _delete_session_file(self, session_id: str) -> bool:
        """
        删除会话文件

        Args:
            session_id: 会话 ID

        Returns:
            删除成功返回 True，否则返回 False
        """
        file_path = self._get_session_file(session_id)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except IOError as e:
                print(f"删除会话文件 {session_id} 失败: {e}")
                return False
        return True

    async def create_session(
        self,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        创建新会话

        Args:
            title: 会话标题（可选）
            metadata: 额外元数据（可选）

        Returns:
            新创建的 Session 实例
        """
        async with self._get_lock():
            session = Session(
                title=title or "新会话",
                metadata=metadata or {}
            )

            # 保存到内存缓存
            self._sessions[session.id] = session

            # 持久化到文件
            await self._save_session(session)

            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取会话

        Args:
            session_id: 会话 ID

        Returns:
            Session 实例，如果不存在则返回 None
        """
        async with self._get_lock():
            # 先检查内存缓存
            if session_id in self._sessions:
                return self._sessions[session_id]

            # 从文件加载
            session = await self._load_session(session_id)
            if session:
                self._sessions[session_id] = session

            return session

    async def update_session(self, session: Session) -> bool:
        """
        更新会话

        Args:
            session: 要更新的会话

        Returns:
            更新成功返回 True，否则返回 False
        """
        async with self._get_lock():
            session.updated_at = datetime.now().isoformat()
            self._sessions[session.id] = session
            return await self._save_session(session)

    async def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话 ID

        Returns:
            删除成功返回 True，否则返回 False
        """
        async with self._get_lock():
            # 从内存缓存中删除
            if session_id in self._sessions:
                del self._sessions[session_id]

            # 删除文件
            return await self._delete_session_file(session_id)

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> List[Session]:
        """
        获取会话列表

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            Session 列表
        """
        sessions = []

        # 遍历数据目录中的所有会话文件
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json'):
                    session_id = filename[:-5]  # 移除 .json 后缀
                    session = await self.get_session(session_id)
                    if session:
                        sessions.append(session)
        except IOError:
            pass

        # 按更新时间排序（最新的在前）
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        # 分页
        return sessions[offset:offset + limit]

    async def add_message_to_session(
        self,
        session_id: str,
        message: Message
    ) -> bool:
        """
        向会话添加消息

        Args:
            session_id: 会话 ID
            message: 要添加的消息

        Returns:
            添加成功返回 True，否则返回 False
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        session.add_message(message)
        return await self.update_session(session)

    async def get_session_messages(
        self,
        session_id: str
    ) -> List[Message]:
        """
        获取会话的所有消息

        Args:
            session_id: 会话 ID

        Returns:
            Message 列表
        """
        session = await self.get_session(session_id)
        if not session:
            return []

        return session.get_messages()

    async def clear_session_messages(self, session_id: str) -> bool:
        """
        清空会话消息

        Args:
            session_id: 会话 ID

        Returns:
            清空成功返回 True，否则返回 False
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        session.clear_messages()
        return await self.update_session(session)

    async def cleanup_expired_sessions(self) -> int:
        """
        清理过期会话

        Returns:
            清理的会话数量
        """
        expired_count = 0
        expire_threshold = datetime.now() - timedelta(hours=self.expire_hours)

        sessions = await self.list_sessions(limit=1000)

        for session in sessions:
            updated_at = datetime.fromisoformat(session.updated_at)
            if updated_at < expire_threshold:
                await self.delete_session(session.id)
                expired_count += 1

        return expired_count

    async def get_session_count(self) -> int:
        """
        获取会话总数

        Returns:
            会话数量
        """
        count = 0
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json'):
                    count += 1
        except IOError:
            pass
        return count
