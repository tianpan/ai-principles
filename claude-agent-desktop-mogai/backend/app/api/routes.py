# -*- coding: utf-8 -*-
"""
API Routes - API 路由定义

定义所有 REST API 端点和 SSE 流式端点
"""

import json
import asyncio
from typing import AsyncGenerator, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ..core.config import settings
from ..core.agent_engine import AgentEngine, Message
from ..core.session_manager import SessionManager
from ..core.skills_registry import SkillsRegistry
from ..core.context_manager import ContextManager
from ..models.schemas import (
    # 请求模型
    MessageCreate,
    SessionCreate,
    SessionUpdate,
    SkillExecuteRequest,
    ChatRequest,
    # 响应模型
    MessageResponse,
    SessionResponse,
    SessionDetailResponse,
    SkillInfo,
    SkillExecuteResponse,
    ChatResponse,
    StreamChunk,
    HealthResponse,
    ErrorResponse,
    PaginatedResponse,
    ContextStatsResponse,
    SystemStatsResponse
)


# ==================== 创建路由器 ====================
router = APIRouter()


# ==================== 依赖注入 ====================

# 全局实例（单例模式）
_session_manager: SessionManager = None
_skills_registry: SkillsRegistry = None
_context_manager: ContextManager = None


def get_session_manager() -> SessionManager:
    """获取会话管理器实例"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_skills_registry() -> SkillsRegistry:
    """获取技能注册表实例"""
    global _skills_registry
    if _skills_registry is None:
        _skills_registry = SkillsRegistry()
    return _skills_registry


def get_context_manager() -> ContextManager:
    """获取上下文管理器实例"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


async def get_agent_engine() -> AgentEngine:
    """获取 Agent 引擎实例"""
    if not settings.validate_api_key():
        raise HTTPException(
            status_code=500,
            detail="Anthropic API Key 未配置，请在 .env 文件中设置 ANTHROPIC_API_KEY"
        )
    return AgentEngine(skills_registry=get_skills_registry())


# ==================== 健康检查端点 ====================

@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查端点

    用于检查服务是否正常运行
    """
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        timestamp=datetime.now().isoformat()
    )


@router.get("/stats", response_model=SystemStatsResponse, tags=["系统"])
async def get_system_stats(
    session_manager: SessionManager = Depends(get_session_manager),
    skills_registry: SkillsRegistry = Depends(get_skills_registry)
):
    """
    获取系统统计信息

    返回会话数、技能数等统计数据
    """
    total_sessions = await session_manager.get_session_count()
    total_skills = len(skills_registry.get_all_skills())

    return SystemStatsResponse(
        total_sessions=total_sessions,
        total_skills=total_skills,
        active_sessions=0,  # TODO: 实现活跃会话统计
        uptime_seconds=0,   # TODO: 实现运行时间统计
        version=settings.app_version
    )


# ==================== 会话管理端点 ====================

@router.get("/sessions", response_model=PaginatedResponse[SessionResponse], tags=["会话"])
async def list_sessions(
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    获取会话列表

    支持分页查询
    """
    offset = (page - 1) * page_size
    sessions = await session_manager.list_sessions(limit=page_size, offset=offset)

    # 转换为响应格式
    items = [
        SessionResponse(
            id=s.id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=len(s.messages),
            metadata=s.metadata
        )
        for s in sessions
    ]

    # 获取总数
    total = await session_manager.get_session_count()

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + len(sessions)) < total
    )


@router.post("/sessions", response_model=SessionResponse, tags=["会话"])
async def create_session(
    request: SessionCreate,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    创建新会话

    创建一个新的对话会话
    """
    session = await session_manager.create_session(
        title=request.title,
        metadata=request.metadata
    )

    return SessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=0,
        metadata=session.metadata
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse, tags=["会话"])
async def get_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    获取会话详情

    返回会话信息及消息历史
    """
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = [
        MessageResponse(
            role=m.get("role"),
            content=m.get("content"),
            timestamp=m.get("timestamp"),
            metadata=m.get("metadata")
        )
        for m in session.messages
    ]

    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(session.messages),
        metadata=session.metadata,
        messages=messages
    )


@router.delete("/sessions/{session_id}", tags=["会话"])
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    删除会话

    删除指定的会话及其消息历史
    """
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")

    return {"success": True, "message": "会话已删除"}


@router.delete("/sessions/{session_id}/messages", tags=["会话"])
async def clear_session_messages(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    清空会话消息

    清空指定会话的所有消息历史
    """
    success = await session_manager.clear_session_messages(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")

    return {"success": True, "message": "消息已清空"}


# ==================== 聊天端点 ====================

@router.post("/chat", response_model=ChatResponse, tags=["聊天"])
async def chat(
    request: ChatRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    agent_engine: AgentEngine = Depends(get_agent_engine)
):
    """
    发送聊天消息（非流式）

    发送消息并获取完整回复
    """
    # 获取或创建会话
    if request.session_id:
        session = await session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        session = await session_manager.create_session()

    # 添加用户消息
    user_message = Message(role="user", content=request.message)
    await session_manager.add_message_to_session(session.id, user_message)

    # 获取历史消息
    history = await session_manager.get_session_messages(session.id)

    # 处理消息（非流式）
    full_response = ""
    async for chunk in agent_engine.chat(request.message, history[:-1], stream=False):
        if chunk.get("type") == "text":
            full_response += chunk.get("content", "")
        elif chunk.get("type") == "done":
            break

    # 创建助手消息
    assistant_message = Message(role="assistant", content=full_response)
    await session_manager.add_message_to_session(session.id, assistant_message)

    return ChatResponse(
        session_id=session.id,
        message=MessageResponse(
            role="assistant",
            content=full_response,
            timestamp=assistant_message.timestamp
        ),
        finish_reason="stop"
    )


@router.post("/chat/stream", tags=["聊天"])
async def chat_stream(
    request: ChatRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    agent_engine: AgentEngine = Depends(get_agent_engine)
):
    """
    发送聊天消息（流式）

    使用 Server-Sent Events (SSE) 返回流式响应
    """
    # 获取或创建会话
    if request.session_id:
        session = await session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        session = await session_manager.create_session()

    async def generate() -> AsyncGenerator[str, None]:
        """生成 SSE 事件流"""
        try:
            # 添加用户消息
            user_message = Message(role="user", content=request.message)
            await session_manager.add_message_to_session(session.id, user_message)

            # 发送会话 ID
            yield json.dumps({
                "type": "session_start",
                "session_id": session.id
            })

            # 获取历史消息
            history = await session_manager.get_session_messages(session.id)

            # 处理消息
            full_response = ""
            async for chunk in agent_engine.chat(request.message, history[:-1], stream=True):
                chunk_data = {
                    "type": chunk.get("type"),
                    "session_id": session.id
                }

                if chunk.get("type") == "text":
                    content = chunk.get("content", "")
                    full_response += content
                    chunk_data["content"] = content

                elif chunk.get("type") == "tool_use":
                    chunk_data["tool_name"] = chunk.get("name")
                    chunk_data["tool_arguments"] = chunk.get("arguments")

                elif chunk.get("type") == "tool_result":
                    chunk_data["tool_name"] = chunk.get("name")
                    chunk_data["tool_result"] = chunk.get("result")

                elif chunk.get("type") == "error":
                    chunk_data["error"] = chunk.get("error")

                elif chunk.get("type") == "done":
                    # 保存助手消息
                    assistant_message = Message(role="assistant", content=full_response)
                    await session_manager.add_message_to_session(session.id, assistant_message)
                    chunk_data["finish_reason"] = "stop"

                yield json.dumps(chunk_data)

        except Exception as e:
            yield json.dumps({
                "type": "error",
                "session_id": session.id,
                "error": str(e)
            })

    return EventSourceResponse(generate())


# ==================== 技能端点 ====================

@router.get("/skills", response_model=List[SkillInfo], tags=["技能"])
async def list_skills(
    category: Optional[str] = Query(default=None, description="按分类过滤"),
    skills_registry: SkillsRegistry = Depends(get_skills_registry)
):
    """
    获取技能列表

    返回所有可用技能，支持按分类过滤
    """
    if category:
        skills = skills_registry.get_skills_by_category(category)
    else:
        skills = skills_registry.get_all_skills()

    return [
        SkillInfo(
            name=s.name,
            description=s.description,
            category=s.category,
            parameters=s.parameters,
            metadata=s.metadata
        )
        for s in skills
    ]


@router.get("/skills/categories", response_model=List[str], tags=["技能"])
async def list_skill_categories(
    skills_registry: SkillsRegistry = Depends(get_skills_registry)
):
    """
    获取技能分类列表

    返回所有可用的技能分类
    """
    return skills_registry.get_categories()


@router.get("/skills/{skill_name}", response_model=SkillInfo, tags=["技能"])
async def get_skill(
    skill_name: str,
    skills_registry: SkillsRegistry = Depends(get_skills_registry)
):
    """
    获取技能详情

    返回指定技能的详细信息
    """
    skill = skills_registry.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail="技能不存在")

    return SkillInfo(
        name=skill.name,
        description=skill.description,
        category=skill.category,
        parameters=skill.parameters,
        metadata=skill.metadata
    )


@router.post("/skills/{skill_name}/execute", response_model=SkillExecuteResponse, tags=["技能"])
async def execute_skill(
    skill_name: str,
    request: SkillExecuteRequest,
    skills_registry: SkillsRegistry = Depends(get_skills_registry)
):
    """
    执行技能

    手动执行指定的技能
    """
    skill = skills_registry.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail="技能不存在")

    try:
        result = await skills_registry.execute_skill(skill_name, request.arguments)
        return SkillExecuteResponse(
            skill_name=skill_name,
            success=True,
            result=result
        )
    except Exception as e:
        return SkillExecuteResponse(
            skill_name=skill_name,
            success=False,
            error=str(e)
        )


# ==================== 上下文端点 ====================

@router.get("/sessions/{session_id}/context/stats", response_model=ContextStatsResponse, tags=["会话"])
async def get_context_stats(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    context_manager: ContextManager = Depends(get_context_manager)
):
    """
    获取会话上下文统计

    返回消息数、Token 估算等统计信息
    """
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = session.get_messages()
    stats = context_manager.get_context_stats(messages)

    return ContextStatsResponse(**stats)
