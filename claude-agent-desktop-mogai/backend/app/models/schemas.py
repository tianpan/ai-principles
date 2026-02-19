# -*- coding: utf-8 -*-
"""
Pydantic Schemas - API 数据模型定义

定义所有 API 请求和响应的数据结构
"""

from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime
from pydantic import BaseModel, Field


# ==================== 泛型类型变量 ====================
T = TypeVar('T')


# ==================== 基础响应模型 ====================

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(default=True, description="操作是否成功")
    message: Optional[str] = Field(default=None, description="响应消息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="ok", description="服务状态")
    version: str = Field(default="0.2.0", description="服务版本")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="当前时间"
    )


class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = Field(default=False, description="操作失败")
    error: str = Field(..., description="错误信息")
    error_code: Optional[str] = Field(default=None, description="错误代码")
    details: Optional[Dict[str, Any]] = Field(default=None, description="错误详情")


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应"""
    items: List[T] = Field(default_factory=list, description="数据列表")
    total: int = Field(default=0, description="总数")
    page: int = Field(default=1, description="当前页码")
    page_size: int = Field(default=20, description="每页数量")
    has_more: bool = Field(default=False, description="是否有更多数据")


# ==================== 消息模型 ====================

class MessageCreate(BaseModel):
    """创建消息请求"""
    content: str = Field(..., description="消息内容", min_length=1, max_length=10000)
    role: str = Field(default="user", description="消息角色")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class MessageResponse(BaseModel):
    """消息响应"""
    role: str = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    timestamp: str = Field(..., description="消息时间戳")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


# ==================== 会话模型 ====================

class SessionCreate(BaseModel):
    """创建会话请求"""
    title: Optional[str] = Field(default=None, description="会话标题", max_length=200)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="会话元数据")


class SessionUpdate(BaseModel):
    """更新会话请求"""
    title: Optional[str] = Field(default=None, description="会话标题", max_length=200)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="会话元数据")


class SessionResponse(BaseModel):
    """会话响应"""
    id: str = Field(..., description="会话 ID")
    title: str = Field(..., description="会话标题")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    message_count: int = Field(default=0, description="消息数量")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="会话元数据")


class SessionDetailResponse(SessionResponse):
    """会话详情响应（包含消息）"""
    messages: List[MessageResponse] = Field(default_factory=list, description="消息列表")


# ==================== 技能模型 ====================

class SkillParameterSchema(BaseModel):
    """技能参数 Schema"""
    type: str = Field(default="string", description="参数类型")
    description: Optional[str] = Field(default=None, description="参数描述")
    enum: Optional[List[str]] = Field(default=None, description="枚举值列表")
    default: Optional[Any] = Field(default=None, description="默认值")


class SkillInfo(BaseModel):
    """技能信息"""
    name: str = Field(..., description="技能名称")
    description: str = Field(..., description="技能描述")
    category: str = Field(default="general", description="技能分类")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="参数 Schema")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class SkillExecuteRequest(BaseModel):
    """执行技能请求"""
    arguments: Dict[str, Any] = Field(default_factory=dict, description="技能参数")


class SkillExecuteResponse(BaseModel):
    """执行技能响应"""
    skill_name: str = Field(..., description="技能名称")
    success: bool = Field(default=True, description="执行是否成功")
    result: Optional[Any] = Field(default=None, description="执行结果")
    error: Optional[str] = Field(default=None, description="错误信息")


# ==================== 聊天模型 ====================

class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: Optional[str] = Field(default=None, description="会话 ID（可选，不提供则创建新会话）")
    message: str = Field(..., description="用户消息", min_length=1, max_length=10000)
    stream: bool = Field(default=True, description="是否使用流式响应")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="请求元数据")


class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str = Field(..., description="会话 ID")
    message: MessageResponse = Field(..., description="助手回复消息")
    finish_reason: str = Field(default="stop", description="结束原因")


class StreamChunk(BaseModel):
    """流式响应块"""
    type: str = Field(..., description="块类型：text/tool_use/tool_result/done/error")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    content: Optional[str] = Field(default=None, description="文本内容")
    tool_name: Optional[str] = Field(default=None, description="工具名称")
    tool_arguments: Optional[Dict[str, Any]] = Field(default=None, description="工具参数")
    tool_result: Optional[Any] = Field(default=None, description="工具结果")
    error: Optional[str] = Field(default=None, description="错误信息")
    finish_reason: Optional[str] = Field(default=None, description="结束原因")


# ==================== 统计模型 ====================

class ContextStatsResponse(BaseModel):
    """上下文统计响应"""
    total_messages: int = Field(default=0, description="总消息数")
    user_messages: int = Field(default=0, description="用户消息数")
    assistant_messages: int = Field(default=0, description="助手消息数")
    estimated_tokens: int = Field(default=0, description="估算 Token 数")
    should_compress: bool = Field(default=False, description="是否需要压缩")
    max_messages: int = Field(default=50, description="最大消息数")
    compress_threshold: int = Field(default=30, description="压缩阈值")


class SystemStatsResponse(BaseModel):
    """系统统计响应"""
    total_sessions: int = Field(default=0, description="总会话数")
    total_skills: int = Field(default=0, description="总技能数")
    active_sessions: int = Field(default=0, description="活跃会话数")
    uptime_seconds: float = Field(default=0, description="运行时间（秒）")
    version: str = Field(default="0.2.0", description="系统版本")
