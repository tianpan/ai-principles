"""
测试配置文件 - pytest fixtures
==============================

提供测试所需的各种 fixtures：
- 测试客户端
- 模拟配置
- 测试数据
- Mock 对象

使用方法：
    def test_example(test_client, mock_config):
        response = test_client.get("/api/health")
        assert response.status_code == 200
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# 事件循环配置
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    创建会话级别的事件循环

    用于所有异步测试，确保测试之间的事件循环一致性。
    scope="session" 表示整个测试会话使用同一个事件循环。
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# 配置 Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    模拟配置对象

    提供测试环境所需的配置参数，包括：
    - 数据库配置
    - API 密钥
    - 模型配置
    - 服务端口等
    """
    return {
        # 应用配置
        "app_name": "Towngas Manus Test",
        "debug": True,
        "environment": "testing",
        # API 配置
        "api_prefix": "/api/v1",
        "api_key": "test-api-key-12345",
        # 模型配置
        "model": {
            "provider": "anthropic",
            "name": "claude-3-sonnet-20240229",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        # 数据库配置
        "database": {
            "url": "sqlite:///:memory:",
            "echo": False,
        },
        # Redis 配置
        "redis": {
            "url": "redis://localhost:6379/15",  # 使用 15 号测试数据库
        },
        # 日志配置
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        # 会话配置
        "session": {
            "max_history": 100,
            "timeout": 3600,  # 1 小时
            "storage": "memory",
        },
    }


@pytest.fixture
def mock_settings(mock_config: Dict[str, Any]) -> MagicMock:
    """
    模拟 Settings 对象

    将配置字典转换为属性可访问的对象，模拟 pydantic Settings。
    """
    settings = MagicMock()
    for key, value in mock_config.items():
        setattr(settings, key.upper(), value)
    return settings


# =============================================================================
# 测试客户端 Fixtures
# =============================================================================


@pytest.fixture
def app() -> FastAPI:
    """
    创建测试用 FastAPI 应用实例

    创建一个最小化的 FastAPI 应用，用于测试。
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Towngas Manus Test API",
        version="0.1.0-test",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 添加 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


@pytest.fixture
def test_client(app: FastAPI) -> TestClient:
    """
    同步测试客户端

    用于测试同步 API 端点。
    自动处理请求/响应生命周期。
    """
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """
    异步测试客户端

    用于测试异步 API 端点，如流式响应。
    需要配合 pytest-asyncio 使用。
    """
    from httpx import ASGITransport

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


# =============================================================================
# 模拟数据 Fixtures
# =============================================================================


@pytest.fixture
def sample_message() -> Dict[str, Any]:
    """
    示例消息数据

    提供标准的用户消息格式，用于测试消息处理。
    """
    return {
        "role": "user",
        "content": "你好，请介绍一下港华燃气的业务范围。",
        "timestamp": "2024-01-15T10:30:00Z",
    }


@pytest.fixture
def sample_assistant_message() -> Dict[str, Any]:
    """
    示例助手消息数据

    提供标准的助手响应格式。
    """
    return {
        "role": "assistant",
        "content": "港华燃气是香港领先的燃气供应商，主要业务包括：\n1. 燃气供应\n2. 管道安装维护\n3. 燃气设备销售",
        "timestamp": "2024-01-15T10:30:05Z",
    }


@pytest.fixture
def sample_conversation() -> Dict[str, Any]:
    """
    示例对话数据

    包含多轮对话的完整消息历史。
    """
    return {
        "session_id": "test-session-123",
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
            {"role": "user", "content": "请介绍港华燃气"},
            {"role": "assistant", "content": "港华燃气是..."},
        ],
        "metadata": {
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:30:00Z",
            "model": "claude-3-sonnet",
        },
    }


@pytest.fixture
def sample_skill() -> Dict[str, Any]:
    """
    示例技能数据

    定义一个测试用技能，包含名称、描述和参数。
    """
    return {
        "name": "gas_price_query",
        "description": "查询燃气价格",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "地区名称，如：香港、深圳",
                },
                "usage_type": {
                    "type": "string",
                    "enum": ["residential", "commercial", "industrial"],
                    "description": "用户类型",
                },
            },
            "required": ["region"],
        },
        "handler": "skills.gas_price.handle_query",
    }


@pytest.fixture
def sample_skills_list() -> list[Dict[str, Any]]:
    """
    示例技能列表

    包含多个测试用技能。
    """
    return [
        {
            "name": "gas_price_query",
            "description": "查询燃气价格",
            "category": "inquiry",
        },
        {
            "name": "bill_calculator",
            "description": "计算燃气费用",
            "category": "utility",
        },
        {
            "name": "appointment_booking",
            "description": "预约服务",
            "category": "service",
        },
        {
            "name": "complaint_submit",
            "description": "提交投诉",
            "category": "feedback",
        },
    ]


# =============================================================================
# Mock 对象 Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """
    模拟 LLM 客户端

    模拟 Anthropic/OpenAI 等 LLM API 调用，
    避免在测试中进行真实的 API 请求。
    """

    async def mock_generate(*args, **kwargs):
        return {
            "content": "这是一个模拟的 LLM 响应",
            "model": "claude-3-sonnet",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

    client = AsyncMock()
    client.generate = AsyncMock(side_effect=mock_generate)
    client.stream = AsyncMock()
    return client


@pytest.fixture
def mock_session_store() -> MagicMock:
    """
    模拟会话存储

    提供内存中的会话存储实现，用于测试会话管理。
    """
    store = MagicMock()
    store._sessions = {}  # 内存存储

    def get_session(session_id: str):
        return store._sessions.get(session_id)

    def save_session(session_id: str, data: dict):
        store._sessions[session_id] = data
        return True

    def delete_session(session_id: str):
        if session_id in store._sessions:
            del store._sessions[session_id]
            return True
        return False

    store.get.side_effect = get_session
    store.save.side_effect = save_session
    store.delete.side_effect = delete_session
    store.list.return_value = lambda: list(store._sessions.keys())

    return store


@pytest.fixture
def mock_skill_executor() -> AsyncMock:
    """
    模拟技能执行器

    模拟技能的执行过程，返回预设结果。
    """

    async def execute_skill(skill_name: str, params: dict):
        if skill_name == "gas_price_query":
            return {
                "success": True,
                "result": {
                    "region": params.get("region", "香港"),
                    "price_per_unit": 3.5,
                    "unit": "立方米",
                    "currency": "HKD",
                },
            }
        elif skill_name == "bill_calculator":
            return {
                "success": True,
                "result": {
                    "total": 350.0,
                    "usage": 100,
                    "breakdown": {"base": 300, "tax": 50},
                },
            }
        else:
            return {"success": False, "error": f"Unknown skill: {skill_name}"}

    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=execute_skill)
    return executor


# =============================================================================
# 测试工具函数
# =============================================================================


@pytest.fixture
def assert_response_valid():
    """
    响应验证工具

    用于验证 API 响应格式的正确性。
    """

    def _validate(response, expected_status: int = 200):
        assert response.status_code == expected_status, (
            f"Expected status {expected_status}, got {response.status_code}. "
            f"Response: {response.text}"
        )
        if expected_status == 200:
            json_data = response.json()
            assert "success" in json_data or "data" in json_data or "error" in json_data
        return response.json()

    return _validate


@pytest.fixture
def create_test_file(tmp_path: Path):
    """
    创建测试文件工具

    在临时目录中创建测试文件，测试结束后自动清理。
    """

    def _create(filename: str, content: str) -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create


# =============================================================================
# 清理 Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """
    自动清理测试数据

    在每个测试运行后清理测试产生的数据。
    autouse=True 表示自动应用于所有测试。
    """
    yield
    # 测试后的清理逻辑
    # 例如：清理临时文件、重置状态等


# =============================================================================
# 环境变量 Fixtures
# =============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    模拟环境变量

    使用 monkeypatch 安全地设置测试环境变量，
    测试结束后自动恢复原始值。
    """
    env_vars = {
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "OPENAI_API_KEY": "test-openai-key",
        "DATABASE_URL": "sqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/15",
        "LOG_LEVEL": "DEBUG",
        "ENVIRONMENT": "testing",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars
