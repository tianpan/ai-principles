# -*- coding: utf-8 -*-
"""
Context Manager - 上下文管理模块

负责管理对话上下文，包括：
- 上下文压缩
- 消息摘要
- Token 计数
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .config import settings
from .agent_engine import Message


@dataclass
class ContextSummary:
    """
    上下文摘要数据类

    Attributes:
        summary: 摘要内容
        message_count: 摘要涵盖的消息数
        created_at: 创建时间
        key_topics: 关键主题列表
    """
    summary: str
    message_count: int
    created_at: str
    key_topics: List[str]


class ContextManager:
    """
    上下文管理器

    负责管理对话上下文，包括压缩、摘要等

    Attributes:
        max_messages: 最大消息数
        compress_threshold: 压缩阈值
    """

    def __init__(
        self,
        max_messages: Optional[int] = None,
        compress_threshold: Optional[int] = None
    ):
        """
        初始化上下文管理器

        Args:
            max_messages: 最大消息数（可选，默认使用配置中的值）
            compress_threshold: 压缩阈值（可选，默认使用配置中的值）
        """
        self.max_messages = max_messages or settings.context_max_messages
        self.compress_threshold = compress_threshold or settings.context_compress_threshold

    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的 token 数

        使用简单的估算方法：中文约 1.5 字符/token，英文约 4 字符/token

        Args:
            text: 要估算的文本

        Returns:
            估算的 token 数
        """
        # 分离中文字符和其他字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars

        # 估算 token 数
        tokens = int(chinese_chars / 1.5) + int(other_chars / 4)
        return max(tokens, 1)

    def estimate_messages_tokens(self, messages: List[Message]) -> int:
        """
        估算消息列表的总 token 数

        Args:
            messages: 消息列表

        Returns:
            总 token 数
        """
        total = 0
        for msg in messages:
            total += self.estimate_tokens(msg.content)
            # 每条消息有一些额外的 token 开销
            total += 4
        return total

    def should_compress(self, messages: List[Message]) -> bool:
        """
        判断是否需要压缩

        Args:
            messages: 消息列表

        Returns:
            如果需要压缩返回 True，否则返回 False
        """
        return len(messages) >= self.compress_threshold

    def compress_messages(
        self,
        messages: List[Message],
        keep_recent: int = 10
    ) -> tuple[List[Message], Optional[ContextSummary]]:
        """
        压缩消息列表

        保留最近的消息，对较早的消息生成摘要

        Args:
            messages: 消息列表
            keep_recent: 保留的最近消息数

        Returns:
            (压缩后的消息列表, 上下文摘要)
        """
        if len(messages) <= keep_recent:
            return messages, None

        # 分割消息
        to_compress = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]

        # 生成摘要
        summary = self._generate_summary(to_compress)

        # 创建摘要消息
        summary_message = Message(
            role="assistant",
            content=f"[上下文摘要] {summary.summary}",
            metadata={
                "type": "context_summary",
                "message_count": summary.message_count,
                "key_topics": summary.key_topics
            }
        )

        # 返回压缩后的消息
        compressed = [summary_message] + recent_messages
        return compressed, summary

    def _generate_summary(self, messages: List[Message]) -> ContextSummary:
        """
        生成消息摘要

        Args:
            messages: 要摘要的消息列表

        Returns:
            ContextSummary 实例
        """
        # 提取关键信息
        topics = set()
        user_queries = []
        assistant_responses = []

        for msg in messages:
            if msg.role == "user":
                user_queries.append(msg.content[:100])
                # 提取关键词（简单实现）
                keywords = self._extract_keywords(msg.content)
                topics.update(keywords)
            else:
                assistant_responses.append(msg.content[:100])

        # 生成摘要文本
        summary_parts = []
        if user_queries:
            summary_parts.append(f"用户进行了 {len(user_queries)} 次查询")
        if assistant_responses:
            summary_parts.append(f"助手提供了 {len(assistant_responses)} 次回复")

        summary_text = "，".join(summary_parts)

        # 添加主要话题
        if topics:
            topics_list = list(topics)[:5]  # 最多 5 个话题
            summary_text += f"。主要话题：{', '.join(topics_list)}"

        return ContextSummary(
            summary=summary_text or "无重要内容",
            message_count=len(messages),
            created_at=datetime.now().isoformat(),
            key_topics=list(topics)[:5]
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取关键词（简单实现）

        Args:
            text: 文本内容

        Returns:
            关键词列表
        """
        # 简单的关键词提取：提取中文词组
        # 实际应用中可以使用 NLP 库进行更精确的提取
        keywords = []

        # 匹配 2-4 个中文字符的词组
        pattern = r'[\u4e00-\u9fff]{2,4}'
        matches = re.findall(pattern, text)

        # 过滤常见停用词
        stopwords = {'你好', '请问', '谢谢', '感谢', '可以', '能够', '需要', '是否'}

        for match in matches:
            if match not in stopwords and len(match) >= 2:
                keywords.append(match)

        # 去重并返回
        return list(set(keywords))[:10]

    def truncate_message(
        self,
        message: Message,
        max_tokens: int = 2000
    ) -> Message:
        """
        截断消息以控制 token 数

        Args:
            message: 要截断的消息
            max_tokens: 最大 token 数

        Returns:
            截断后的消息
        """
        current_tokens = self.estimate_tokens(message.content)

        if current_tokens <= max_tokens:
            return message

        # 估算需要保留的字符数
        ratio = max_tokens / current_tokens
        keep_chars = int(len(message.content) * ratio)

        # 截断内容
        truncated_content = message.content[:keep_chars]
        truncated_content += "\n...[内容已截断]"

        return Message(
            role=message.role,
            content=truncated_content,
            timestamp=message.timestamp,
            metadata={
                **message.metadata,
                "truncated": True,
                "original_tokens": current_tokens
            }
        )

    def prepare_messages_for_api(
        self,
        messages: List[Message],
        max_tokens: int = 100000
    ) -> List[Message]:
        """
        准备发送给 API 的消息

        包括压缩、截断等处理，确保不超过 token 限制

        Args:
            messages: 消息列表
            max_tokens: 最大 token 限制

        Returns:
            处理后的消息列表
        """
        result = list(messages)

        # 检查是否需要压缩
        while self.estimate_messages_tokens(result) > max_tokens:
            if len(result) <= 2:
                # 只剩系统消息，截断最后一条
                result[-1] = self.truncate_message(result[-1], max_tokens - 100)
                break

            # 尝试压缩
            result, _ = self.compress_messages(result, keep_recent=2)

        return result

    def get_context_stats(self, messages: List[Message]) -> Dict[str, Any]:
        """
        获取上下文统计信息

        Args:
            messages: 消息列表

        Returns:
            统计信息字典
        """
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]

        return {
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "estimated_tokens": self.estimate_messages_tokens(messages),
            "should_compress": self.should_compress(messages),
            "max_messages": self.max_messages,
            "compress_threshold": self.compress_threshold
        }
