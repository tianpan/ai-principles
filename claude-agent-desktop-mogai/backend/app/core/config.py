# -*- coding: utf-8 -*-
"""
Configuration Module - 配置管理模块

负责加载和管理应用配置，包括：
- 环境变量加载
- API 密钥管理
- 服务配置
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    应用配置类

    使用 Pydantic Settings 管理配置，支持从环境变量和 .env 文件加载

    Attributes:
        app_name: 应用名称
        app_version: 应用版本
        debug: 调试模式
        api_host: API 服务主机
        api_port: API 服务端口

        anthropic_api_key: Anthropic API 密钥（必填）
        anthropic_model: 默认使用的 Claude 模型
        anthropic_max_tokens: 最大生成 token 数

        data_dir: 数据存储目录
        session_expire_hours: 会话过期时间（小时）

        cors_origins: CORS 允许的来源列表
    """

    # ==================== 应用基础配置 ====================
    app_name: str = Field(default="Towngas Manus", description="应用名称")
    app_version: str = Field(default="0.2.0", description="应用版本")
    debug: bool = Field(default=False, description="调试模式")
    api_host: str = Field(default="0.0.0.0", description="API 服务主机")
    api_port: int = Field(default=8000, description="API 服务端口")

    # ==================== Anthropic API 配置 ====================
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API 密钥（必填，从 https://console.anthropic.com 获取）"
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="默认使用的 Claude 模型"
    )
    anthropic_max_tokens: int = Field(
        default=4096,
        description="最大生成 token 数"
    )
    anthropic_base_url: Optional[str] = Field(
        default=None,
        description="API 基础 URL（可选，用于代理或自定义端点）"
    )

    # ==================== 数据存储配置 ====================
    data_dir: str = Field(
        default="data",
        description="数据存储目录（相对于 backend/ 目录）"
    )
    session_expire_hours: int = Field(
        default=24,
        description="会话过期时间（小时）"
    )

    # ==================== 上下文管理配置 ====================
    context_max_messages: int = Field(
        default=50,
        description="上下文最大消息数"
    )
    context_compress_threshold: int = Field(
        default=30,
        description="上下文压缩阈值（消息数）"
    )

    # ==================== CORS 配置 ====================
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
        description="CORS 允许的来源列表"
    )

    # ==================== TOP 系统集成配置（港华专用） ====================
    top_api_base_url: Optional[str] = Field(
        default=None,
        description="TOP 系统 API 基础 URL"
    )
    top_api_key: Optional[str] = Field(
        default=None,
        description="TOP 系统 API 密钥"
    )

    # ==================== 知识库配置 ====================
    knowledge_base_path: Optional[str] = Field(
        default=None,
        description="知识库文件路径"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="嵌入模型名称"
    )

    class Config:
        """Pydantic Settings 配置"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_data_path(self, filename: str = "") -> str:
        """
        获取数据文件的完整路径

        Args:
            filename: 文件名（可选）

        Returns:
            完整的文件路径
        """
        # 获取 backend 目录的绝对路径
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(backend_dir, self.data_dir)

        # 确保数据目录存在
        os.makedirs(data_path, exist_ok=True)

        if filename:
            return os.path.join(data_path, filename)
        return data_path

    def validate_api_key(self) -> bool:
        """
        验证 API 密钥是否已配置

        Returns:
            如果 API 密钥已配置返回 True，否则返回 False
        """
        return bool(self.anthropic_api_key and len(self.anthropic_api_key) > 10)


# 创建全局配置实例
settings = Settings()
