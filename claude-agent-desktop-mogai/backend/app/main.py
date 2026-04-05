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
import time
from collections import defaultdict
from threading import Lock

# 确保能找到 app 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.api.routes import router as api_router


# ==================== 速率限制中间件 ====================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    简单的 IP 速率限制中间件

    限制每个 IP 地址在一定时间窗口内的请求数量
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour

        # 存储 IP 访问记录：{ip: [(timestamp, path), ...]}
        self._requests = defaultdict(list)
        self._lock = Lock()

    def _get_client_ip(self, request: Request) -> str:
        """获取客户端 IP 地址"""
        # 检查代理头部
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # 直接连接
        if request.client:
            return request.client.host

        return "unknown"

    def _cleanup_old_requests(self, ip: str, current_time: float):
        """清理过期的请求记录"""
        hour_ago = current_time - 3600
        self._requests[ip] = [
            (ts, path) for ts, path in self._requests[ip]
            if ts > hour_ago
        ]

    def _is_rate_limited(self, ip: str, current_time: float) -> tuple:
        """
        检查是否超过速率限制

        Returns:
            (is_limited, retry_after_seconds)
        """
        with self._lock:
            self._cleanup_old_requests(ip, current_time)

            # 计算每分钟请求数
            minute_ago = current_time - 60
            requests_last_minute = sum(
                1 for ts, _ in self._requests[ip] if ts > minute_ago
            )

            # 计算每小时请求数
            requests_last_hour = len(self._requests[ip])

            if requests_last_minute >= self.requests_per_minute:
                return True, 60 - (current_time - min(ts for ts, _ in self._requests[ip] if ts > minute_ago))

            if requests_last_hour >= self.requests_per_hour:
                return True, 3600 - (current_time - min(ts for ts, _ in self._requests[ip]))

            return False, 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求"""
        # 跳过健康检查和静态资源
        if request.url.path in ["/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        current_time = time.time()

        is_limited, retry_after = self._is_rate_limited(client_ip, current_time)

        if is_limited:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "请求过于频繁，请稍后重试",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": int(retry_after)
                },
                headers={"Retry-After": str(int(retry_after))}
            )

        # 记录请求
        with self._lock:
            self._requests[client_ip].append((current_time, request.url.path))

        return await call_next(request)


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

# 根据 DEBUG 模式调整 CORS 严格程度
if settings.debug:
    # 开发环境：宽松配置
    cors_allow_methods = ["*"]
    cors_allow_headers = ["*"]
else:
    # 生产环境：严格配置
    cors_allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers = ["Content-Type", "Authorization", "X-Requested-With"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=cors_allow_methods,
    allow_headers=cors_allow_headers,
)

# 添加速率限制中间件（生产环境启用）
if not settings.debug:
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=60,
        requests_per_hour=1000
    )


# ==================== 全局异常处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    # 在生产环境中记录详细错误到日志，但只返回通用错误给客户端
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Internal error: {exc}", exc_info=True)

    # 根据调试模式决定返回详细程度
    if settings.debug:
        error_detail = str(exc)
    else:
        error_detail = "服务器内部错误，请稍后重试"

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": error_detail,
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
