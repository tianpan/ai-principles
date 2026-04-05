# -*- coding: utf-8 -*-
"""
Test Configuration - pytest fixtures
====================================

Provides fixtures for testing:
- Test clients
- Mock configurations
- Test data
- Mock objects for external dependencies

Usage:
    def test_example(test_client, mock_config):
        response = test_client.get("/api/health")
        assert response.status_code == 200
"""

import asyncio
import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create session-level event loop

    Used for all async tests to ensure event loop consistency between tests.
    scope="session" means the entire test session uses the same event loop.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    Mock configuration object

    Provides configuration parameters needed for the test environment, including:
    - Database configuration
    - API keys
    - Model configuration
    - Service ports, etc.
    """
    return {
        # Application configuration
        "app_name": "Towngas Manus Test",
        "debug": True,
        "environment": "testing",
        # API configuration
        "api_prefix": "/api/v1",
        "api_key": "test-api-key-12345",
        # Model configuration
        "model": {
            "provider": "anthropic",
            "name": "claude-3-sonnet-20240229",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        # Database configuration
        "database": {
            "url": "sqlite:///:memory:",
            "echo": False,
        },
        # Redis configuration
        "redis": {
            "url": "redis://localhost:6379/15",  # Use test database 15
        },
        # Logging configuration
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        # Session configuration
        "session": {
            "max_history": 100,
            "timeout": 3600,  # 1 hour
            "storage": "memory",
        },
    }


@pytest.fixture
def mock_settings(mock_config: Dict[str, Any]) -> MagicMock:
    """
    Mock Settings object

    Converts configuration dictionary to attribute-accessible object, simulating pydantic Settings.
    """
    settings = MagicMock()
    for key, value in mock_config.items():
        setattr(settings, key.upper(), value)
    return settings


# =============================================================================
# Test Client Fixtures
# =============================================================================


@pytest.fixture
def app() -> FastAPI:
    """
    Create test FastAPI application instance

    Creates a minimal FastAPI application for testing.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Towngas Manus Test API",
        version="0.1.0-test",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
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
    Synchronous test client

    Used for testing synchronous API endpoints.
    Automatically handles request/response lifecycle.
    """
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """
    Asynchronous test client

    Used for testing asynchronous API endpoints, such as streaming responses.
    Needs to be used with pytest-asyncio.
    """
    from httpx import ASGITransport

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_message() -> Dict[str, Any]:
    """
    Sample message data

    Provides standard user message format for testing message processing.
    """
    return {
        "role": "user",
        "content": "Hello, please introduce Towngas Gas's business scope.",
        "timestamp": "2024-01-15T10:30:00Z",
    }


@pytest.fixture
def sample_assistant_message() -> Dict[str, Any]:
    """
    Sample assistant message data

    Provides standard assistant response format.
    """
    return {
        "role": "assistant",
        "content": "Towngas Gas is Hong Kong's leading gas supplier, main businesses include:\n1. Gas supply\n2. Pipeline installation and maintenance\n3. Gas equipment sales",
        "timestamp": "2024-01-15T10:30:05Z",
    }


@pytest.fixture
def sample_conversation() -> Dict[str, Any]:
    """
    Sample conversation data

    Contains complete message history for multiple rounds of dialogue.
    """
    return {
        "session_id": "test-session-123",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "Please introduce Towngas Gas"},
            {"role": "assistant", "content": "Towngas Gas is..."},
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
    Sample skill data

    Defines a test skill with name, description and parameters.
    """
    return {
        "name": "gas_price_query",
        "description": "Query gas prices",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Region name, e.g.: Hong Kong, Shenzhen",
                },
                "usage_type": {
                    "type": "string",
                    "enum": ["residential", "commercial", "industrial"],
                    "description": "User type",
                },
            },
            "required": ["region"],
        },
        "handler": "skills.gas_price.handle_query",
    }


@pytest.fixture
def sample_skills_list() -> List[Dict[str, Any]]:
    """
    Sample skills list

    Contains multiple test skills.
    """
    return [
        {
            "name": "gas_price_query",
            "description": "Query gas prices",
            "category": "inquiry",
        },
        {
            "name": "bill_calculator",
            "description": "Calculate gas fees",
            "category": "utility",
        },
        {
            "name": "appointment_booking",
            "description": "Book services",
            "category": "service",
        },
        {
            "name": "complaint_submit",
            "description": "Submit complaints",
            "category": "feedback",
        },
    ]


@pytest.fixture
def sample_input_schema() -> Dict[str, Any]:
    """Sample input schema for tools"""
    return {
        "type": "object",
        "properties": {
            "station_id": {
                "type": "string",
                "description": "Station ID",
            },
            "station_name": {
                "type": "string",
                "description": "Station name (fuzzy match)",
            },
        },
    }


@pytest.fixture
def sample_tool_definition() -> "MCPToolDefinition":
    """Sample MCPToolDefinition instance"""
    from app.mcp.adapter import MCPToolDefinition

    return MCPToolDefinition(
        name="query_station",
        mcp_name="mcp__top__query_station",
        description="Query station information",
        input_schema={
            "type": "object",
            "properties": {
                "station_id": {"type": "string"}
            }
        },
        server_name="top",
    )


# =============================================================================
# Mock Object Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """
    Mock LLM client

    Simulates Anthropic/OpenAI etc. LLM API calls,
    avoiding real API requests during testing.
    """

    async def mock_generate(*args, **kwargs):
        return {
            "content": "This is a simulated LLM response",
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
    Mock session store

    Provides in-memory session storage implementation for testing session management.
    """
    store = MagicMock()
    store._sessions = {}  # Memory storage

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
    Mock skill executor

    Simulates the skill execution process, returning preset results.
    """

    async def execute_skill(skill_name: str, params: dict):
        if skill_name == "gas_price_query":
            return {
                "success": True,
                "result": {
                    "region": params.get("region", "Hong Kong"),
                    "price_per_unit": 3.5,
                    "unit": "cubic meter",
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
# Real Source Code Fixtures (with mocked external dependencies)
# =============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_anthropic_client():
    """
    Mock AsyncAnthropic client for testing

    Returns a mock that can be used to patch AsyncAnthropic
    """
    mock_client = AsyncMock()

    # Mock messages.create for non-streaming
    mock_message = MagicMock()
    mock_message.content = [MagicMock(type="text", text="Test response")]
    mock_message.stop_reason = "end_turn"
    mock_message.model = "claude-3-sonnet"
    mock_message.usage = MagicMock(input_tokens=10, output_tokens=5)

    mock_client.messages.create = AsyncMock(return_value=mock_message)

    # Mock messages.stream for streaming
    mock_stream_manager = AsyncMock()
    mock_stream_manager.__aenter__ = AsyncMock(return_value=mock_stream_manager)
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)
    mock_stream_manager.text_stream = AsyncMock()

    async def mock_text_stream():
        for text in ["Test ", "streaming ", "response"]:
            yield text

    mock_stream_manager.text_stream = mock_text_stream()
    mock_stream_manager.get_final_message = AsyncMock(return_value=mock_message)

    mock_client.messages.stream = MagicMock(return_value=mock_stream_manager)
    mock_client.close = AsyncMock()

    return mock_client


@pytest.fixture
def mock_settings_instance():
    """
    Mock Settings instance for app.core.config
    """
    mock_settings = MagicMock()
    mock_settings.anthropic_api_key = "test-api-key"
    mock_settings.anthropic_model = "claude-3-sonnet-20240229"
    mock_settings.anthropic_max_tokens = 4096
    mock_settings.anthropic_base_url = None
    mock_settings.context_max_messages = 50
    mock_settings.context_compress_threshold = 30
    mock_settings.session_expire_hours = 24
    mock_settings.log_level = "DEBUG"

    def mock_get_data_path(subdir: str):
        import tempfile
        tmpdir = tempfile.mkdtemp()
        return os.path.join(tmpdir, subdir)

    mock_settings.get_data_path = mock_get_data_path

    return mock_settings


@pytest.fixture
def skills_registry():
    """
    Create a real SkillsRegistry instance for testing
    """
    from app.core.skills_registry import SkillsRegistry
    return SkillsRegistry()


@pytest.fixture
def mcp_registry():
    """
    Create a real MCPToolRegistry instance for testing
    """
    from app.mcp import MCPToolRegistry
    return MCPToolRegistry()


@pytest.fixture
def mcp_executor():
    """
    Create a real MCPToolExecutor instance for testing
    """
    from app.mcp.executor import MCPToolExecutor
    return MCPToolExecutor(
        timeout_seconds=5.0,
        max_retries=1,
        enable_cache=False,
    )


@pytest.fixture
def context_manager(mock_settings_instance):
    """
    Create a real ContextManager instance for testing
    """
    from app.core.context_manager import ContextManager
    with patch('app.core.context_manager.settings', mock_settings_instance):
        return ContextManager(
            max_messages=50,
            compress_threshold=30
        )


@pytest.fixture
def session_manager(temp_data_dir, mock_settings_instance):
    """
    Create a real SessionManager instance for testing
    """
    from app.core.session_manager import SessionManager
    with patch('app.core.session_manager.settings', mock_settings_instance):
        mock_settings_instance.get_data_path = lambda x: temp_data_dir
        return SessionManager(
            data_dir=temp_data_dir,
            expire_hours=24
        )


@pytest.fixture
def agent_engine(mock_anthropic_client, skills_registry, mcp_registry, mock_settings_instance):
    """
    Create a real AgentEngine instance with mocked Anthropic client
    """
    from app.core.agent_engine import AgentEngine

    with patch('app.core.agent_engine.AsyncAnthropic', return_value=mock_anthropic_client), \
         patch('app.core.agent_engine.settings', mock_settings_instance):
        engine = AgentEngine(
            api_key="test-api-key",
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            skills_registry=skills_registry,
            mcp_registry=mcp_registry,
            enable_mcp=True,
        )
        engine._mock_client = mock_anthropic_client  # Keep reference for assertions
        yield engine


# =============================================================================
# Test Utility Functions
# =============================================================================


@pytest.fixture
def assert_response_valid():
    """
    Response validation tool

    Used to verify the correctness of API response format.
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
    Create test file tool

    Creates test files in a temporary directory, automatically cleaned up after tests end.
    """

    def _create(filename: str, content: str) -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """
    Automatically cleanup test data

    Cleans up data produced by tests after each test runs.
    autouse=True means it's automatically applied to all tests.
    """
    yield
    # Post-test cleanup logic
    # e.g.: cleanup temp files, reset state, etc.


# =============================================================================
# Environment Variable Fixtures
# =============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Mock environment variables

    Uses monkeypatch to safely set test environment variables,
    automatically restores original values after tests end.
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


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "mcp: mark test as MCP-related")
