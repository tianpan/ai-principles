# -*- coding: utf-8 -*-
"""
Agent Engine - Agent 执行引擎

基于 Claude Agent SDK 构建的 Agent 执行引擎，负责：
- 与 Claude API 交互
- 流式输出处理
- 技能调用
- 上下文管理
- MCP 工具集成
"""

import json
import asyncio
import logging
from typing import AsyncGenerator, Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

import anthropic
from anthropic import AsyncAnthropic

from .config import settings
from .skills_registry import SkillsRegistry
from ..mcp import MCPToolRegistry, MCPToolAdapter

# 配置模块级日志
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    消息数据类

    Attributes:
        role: 消息角色（user/assistant）
        content: 消息内容
        timestamp: 消息时间戳
        metadata: 额外元数据
    """
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_api_format(self) -> Dict[str, str]:
        """
        转换为 Claude API 格式

        Returns:
            符合 Claude API 要求的消息字典
        """
        return {
            "role": self.role,
            "content": self.content
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式（用于序列化）

        Returns:
            包含所有属性的字典
        """
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        从字典创建消息实例

        Args:
            data: 消息字典

        Returns:
            Message 实例
        """
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )


@dataclass
class ToolCall:
    """
    工具调用数据类

    Attributes:
        name: 工具名称
        arguments: 工具参数
        result: 工具执行结果
        error: 错误信息（如果有）
    """
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None


class AgentEngine:
    """
    Agent 执行引擎

    负责与 Claude API 交互，处理消息，执行技能，管理上下文

    Attributes:
        client: AsyncAnthropic 客户端
        model: 使用的模型名称
        max_tokens: 最大生成 token 数
        skills_registry: 技能注册表
        mcp_registry: MCP 工具注册表
        system_prompt: 系统提示词
        enable_mcp: 是否启用 MCP 工具
    """

    # 默认系统提示词
    DEFAULT_SYSTEM_PROMPT = """你是 Towngas Manus，港华集团的智能助手。

## 你的能力
1. 回答关于港华燃气业务的问题
2. 查询场站、设备、管网信息
3. 生成运营日报和报表
4. 提供应急处置建议
5. 执行各种技能工具

## 行为准则
1. 专业、准确、友好
2. 使用中文回复
3. 对于不确定的问题，坦诚说明
4. 主动提供帮助和建议

## 技能使用
当用户请求需要使用技能时，你会调用相应的技能工具来完成任务。
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        skills_registry: Optional[SkillsRegistry] = None,
        mcp_registry: Optional[MCPToolRegistry] = None,
        system_prompt: Optional[str] = None,
        enable_mcp: bool = True,
    ):
        """
        初始化 Agent Engine

        Args:
            api_key: Anthropic API 密钥（可选，默认使用配置中的密钥）
            model: 模型名称（可选，默认使用配置中的模型）
            max_tokens: 最大生成 token 数（可选，默认使用配置中的值）
            skills_registry: 技能注册表（可选，默认创建新实例）
            mcp_registry: MCP 工具注册表（可选，默认创建新实例）
            system_prompt: 系统提示词（可选）
            enable_mcp: 是否启用 MCP 工具（默认 True）
        """
        # 使用传入的参数或配置中的默认值
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.anthropic_model
        self.max_tokens = max_tokens or settings.anthropic_max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # 初始化 Anthropic 客户端
        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if settings.anthropic_base_url:
            client_kwargs["base_url"] = settings.anthropic_base_url

        self.client = AsyncAnthropic(**client_kwargs)

        # 技能注册表
        self.skills_registry = skills_registry or SkillsRegistry()

        # MCP 工具注册表
        self.mcp_registry = mcp_registry or MCPToolRegistry()
        self.enable_mcp = enable_mcp

        # 回调函数
        self._on_tool_call: Optional[Callable[[ToolCall], None]] = None

        logger.info(
            f"AgentEngine initialized: model={self.model}, "
            f"enable_mcp={enable_mcp}, "
            f"skills_count={len(self.skills_registry.get_all_skills())}"
        )

    def set_tool_call_callback(self, callback: Callable[[ToolCall], None]) -> None:
        """
        设置工具调用回调函数

        Args:
            callback: 回调函数，接收 ToolCall 对象
        """
        self._on_tool_call = callback

    def _get_tools_definition(self) -> List[Dict[str, Any]]:
        """
        获取工具定义列表

        整合 Skills 和 MCP Tools，返回 Claude API 格式的工具定义。

        合并顺序：
        1. 传统 Skills（来自 skills_registry）
        2. MCP Tools（来自 mcp_registry，如果启用）

        注意：如果存在同名工具，后添加的 MCP Tools 会覆盖 Skills。
        建议使用不同的命名空间避免冲突（MCP 工具使用 mcp__ 前缀）。

        Returns:
            Claude API 格式的工具定义列表
        """
        tools: List[Dict[str, Any]] = []

        # 1. 添加传统 Skills
        for skill in self.skills_registry.get_all_skills():
            tool_def = {
                "name": skill.name,
                "description": skill.description,
                "input_schema": skill.parameters
            }
            tools.append(tool_def)

        # 2. 添加 MCP Tools（如果启用）
        if self.enable_mcp and self.mcp_registry:
            mcp_tools = self.mcp_registry.get_claude_tools()
            tools.extend(mcp_tools)
            logger.debug(f"Added {len(mcp_tools)} MCP tools to tool definitions")

        logger.debug(f"Total tools available: {len(tools)}")
        return tools

    async def process_message(
        self,
        messages: List[Message],
        stream: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        处理消息并生成回复

        Args:
            messages: 消息历史列表
            stream: 是否使用流式输出

        Yields:
            包含响应内容的字典，格式为：
            - {"type": "text", "content": "..."} - 文本内容
            - {"type": "tool_use", "name": "...", "arguments": {...}} - 工具调用
            - {"type": "tool_result", "name": "...", "result": ...} - 工具结果
            - {"type": "done", "message": {...}} - 完成信号
            - {"type": "error", "error": "..."} - 错误信息
        """
        try:
            # 转换消息格式
            api_messages = [msg.to_api_format() for msg in messages]

            # 获取工具定义
            tools = self._get_tools_definition()

            # 构建 API 请求参数
            request_params: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": self.system_prompt,
                "messages": api_messages,
            }

            # 只有有工具时才添加 tools 参数
            if tools:
                request_params["tools"] = tools

            logger.debug(
                f"Processing message: model={self.model}, "
                f"tools_count={len(tools)}, stream={stream}"
            )

            if stream:
                # 流式输出 - 支持 Agentic Loop（工具调用循环）
                async for chunk in self._process_streaming(request_params):
                    yield chunk
            else:
                # 非流式输出 - 支持 Agentic Loop（工具调用循环）
                async for chunk in self._process_non_streaming(request_params):
                    yield chunk

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            yield {"type": "error", "error": f"API 错误: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error processing message: {e}", exc_info=True)
            yield {"type": "error", "error": f"处理消息时发生错误: {str(e)}"}

    async def _process_streaming(
        self,
        request_params: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理消息（内部方法）

        Args:
            request_params: API 请求参数

        Yields:
            响应内容块
        """
        current_messages = list(request_params["messages"])

        while True:
            logger.debug("Starting streaming request")
            async with self.client.messages.stream(**request_params) as stream_response:
                # 收集流式文本
                async for text in stream_response.text_stream:
                    yield {
                        "type": "text",
                        "content": text
                    }

            # 获取最终消息以检查是否有工具调用
            final_message = await stream_response.get_final_message()
            logger.debug(
                f"Streaming response: stop_reason={final_message.stop_reason}, "
                f"blocks={len(final_message.content)}"
            )

            # 检查是否有工具调用
            tool_calls = [b for b in final_message.content if b.type == "tool_use"]

            if not tool_calls:
                # 没有工具调用，结束循环
                break

            # 执行所有工具调用并收集结果
            assistant_content: List[Dict[str, Any]] = []
            tool_result_content: List[Dict[str, Any]] = []

            for block in final_message.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input

                    logger.info(f"Tool call (streaming): name={tool_name}")

                    yield {
                        "type": "tool_use",
                        "name": tool_name,
                        "arguments": tool_args
                    }

                    # 执行工具
                    result = await self._execute_tool(tool_name, tool_args)

                    yield {
                        "type": "tool_result",
                        "name": tool_name,
                        "result": result
                    }

                    # 收集工具调用和结果用于继续对话
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": tool_name,
                        "input": tool_args
                    })
                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })

            # 添加消息继续对话
            current_messages.append({"role": "assistant", "content": assistant_content})
            current_messages.append({"role": "user", "content": tool_result_content})

            # 更新请求参数
            request_params["messages"] = current_messages

        # 流结束时发送 done 信号
        yield {"type": "done", "message": {}}

    async def _process_non_streaming(
        self,
        request_params: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        非流式处理消息（内部方法）

        Args:
            request_params: API 请求参数

        Yields:
            响应内容块
        """
        full_content = ""
        current_messages = list(request_params["messages"])

        while True:
            response = await self.client.messages.create(**request_params)

            logger.debug(
                f"Response: stop_reason={response.stop_reason}, "
                f"content_blocks={len(response.content)}"
            )

            # 收集工具调用信息和结果
            assistant_content: List[Dict[str, Any]] = []
            tool_result_content: List[Dict[str, Any]] = []
            has_tool_use = False

            for block in response.content:
                if block.type == "text":
                    full_content += block.text
                    yield {
                        "type": "text",
                        "content": block.text
                    }
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    has_tool_use = True
                    tool_name = block.name
                    tool_args = block.input

                    logger.info(f"Tool call (non-streaming): name={tool_name}")

                    yield {
                        "type": "tool_use",
                        "name": tool_name,
                        "arguments": tool_args
                    }

                    # 执行工具（只执行一次）
                    result = await self._execute_tool(tool_name, tool_args)

                    yield {
                        "type": "tool_result",
                        "name": tool_name,
                        "result": result
                    }

                    # 收集工具调用信息
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": tool_name,
                        "input": tool_args
                    })

                    # 收集工具结果（复用已执行的结果）
                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })

            # 如果没有工具调用，结束循环
            if not has_tool_use:
                break

            # 添加助手消息（包含工具调用）和用户消息（包含工具结果）
            current_messages.append({"role": "assistant", "content": assistant_content})
            current_messages.append({"role": "user", "content": tool_result_content})

            # 更新请求参数继续对话
            request_params["messages"] = current_messages

        yield {
            "type": "done",
            "message": {
                "role": "assistant",
                "content": full_content
            }
        }

    async def _execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        执行工具（技能或 MCP 工具）

        根据工具名称判断是 Skills 还是 MCP Tools，路由到相应的执行器。
        MCP 工具通过名称前缀 "mcp__" 进行识别。

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果。如果执行失败，返回 {"error": "错误信息"}
        """
        tool_call = ToolCall(name=name, arguments=arguments)

        try:
            # 判断是否为 MCP 工具（通过 mcp__ 前缀识别）
            if self.enable_mcp and MCPToolAdapter.is_mcp_tool(name):
                logger.debug(f"Routing to MCP tool: {name}")
                result = await self._execute_mcp_tool(name, arguments)
            else:
                logger.debug(f"Routing to Skill: {name}")
                result = await self.skills_registry.execute_skill(name, arguments)

            tool_call.result = result

            # 触发回调
            if self._on_tool_call:
                self._on_tool_call(tool_call)

            return result

        except Exception as e:
            error_msg = str(e)
            tool_call.error = error_msg

            # 记录详细的错误日志
            logger.error(
                f"Tool execution failed: name={name}, error={error_msg}",
                exc_info=True
            )

            # 触发回调（包含错误信息）
            if self._on_tool_call:
                self._on_tool_call(tool_call)

            return {"error": error_msg}

    async def _execute_mcp_tool(
        self,
        mcp_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        执行 MCP 工具

        通过 MCPToolRegistry 执行工具，处理执行结果。

        Args:
            mcp_name: MCP 格式的工具名称（如 "mcp__top__query_station"）
            arguments: 工具参数

        Returns:
            工具执行结果。如果 registry 未初始化或执行失败，
            返回 {"error": "错误信息"}
        """
        if not self.mcp_registry:
            error_msg = "MCP registry not initialized"
            logger.error(error_msg)
            return {"error": error_msg}

        result = await self.mcp_registry.execute_tool(mcp_name, arguments)

        if result.success:
            logger.debug(
                f"MCP tool executed successfully: {mcp_name}, "
                f"time_ms={result.execution_time_ms}"
            )
            return result.result
        else:
            logger.error(
                f"MCP tool execution failed: {mcp_name}, error={result.error}"
            )
            return {"error": result.error}

    async def chat(
        self,
        user_message: str,
        history: List[Message],
        stream: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        便捷的聊天接口

        Args:
            user_message: 用户消息
            history: 历史消息列表
            stream: 是否使用流式输出

        Yields:
            响应内容
        """
        # 创建用户消息
        message = Message(role="user", content=user_message)

        # 合并历史消息
        all_messages = history + [message]

        # 处理消息
        async for chunk in self.process_message(all_messages, stream=stream):
            yield chunk

    async def close(self) -> None:
        """关闭客户端连接，释放资源"""
        await self.client.close()
        logger.info("AgentEngine client closed")
