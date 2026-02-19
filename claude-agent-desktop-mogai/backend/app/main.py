# -*- coding: utf-8 -*-
"""
Towngas Manus Backend - FastAPI 主应用

港华智能体平台后端服务

基于 Claude Agent SDK 构建，提供：
- Agent 执行引擎
- 会话管理
- 技能注册与执行
- RESTful API
- SSE 流式响应
"""

import sys
import os

# 确保能找到 app 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html

from app.core.config import settings
from app.api.routes import router as api_router


# ==================== 应用生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    在启动时执行初始化，在关闭时执行清理
    """
    # 启动时的初始化
    print(f"🚀 {settings.app_name} v{settings.app_version} 启动中...")
    print(f"📅 启动时间: {datetime.now().isoformat()}")

    # 验证 API Key
    if not settings.validate_api_key():
        print("⚠️  警告: ANTHROPIC_API_KEY 未配置或无效")
        print("   请在 .env 文件中设置有效的 API Key")
    else:
        print("✅ API Key 验证通过")

    # 确保数据目录存在
    os.makedirs(settings.get_data_path(), exist_ok=True)
    print(f"📁 数据目录: {settings.get_data_path()}")

    yield

    # 关闭时的清理
    print(f"👋 {settings.app_name} 正在关闭...")


# ==================== 创建 FastAPI 应用 ====================

app = FastAPI(
    title=settings.app_name,
    description="""
## Towngas Manus - 港华智能体平台 API

基于 Claude Agent SDK 构建的企业级 Agent 平台

### 核心功能
- **会话管理**: 创建、查询、删除会话
- **聊天**: 与 Agent 进行对话（支持流式响应）
- **技能系统**: 注册和执行各种技能工具

### 使用方式
1. 创建会话: `POST /api/sessions`
2. 发送消息: `POST /api/chat/stream` (流式) 或 `POST /api/chat` (非流式)
3. 查看技能: `GET /api/skills`
4. 执行技能: `POST /api/skills/{skill_name}/execute`
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# ==================== CORS 配置 ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 全局异常处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "error_code": "INTERNAL_ERROR",
            "timestamp": datetime.now().isoformat()
        }
    )


# ==================== 注册路由 ====================

app.include_router(api_router, prefix="/api")


# ==================== 根路由 ====================

@app.get("/", tags=["根"])
async def root():
    """
    根路由

    返回 API 基本信息
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
        "timestamp": datetime.now().isoformat()
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn

    # 开发环境启动
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
