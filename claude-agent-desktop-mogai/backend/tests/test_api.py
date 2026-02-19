"""
API 集成测试
=============

测试 API 端点功能：
- 聊天 API
- 会话 API
- 技能 API
- 健康检查
- 错误处理

测试覆盖：
- 集成测试：完整 API 流程
- 端到端测试：用户交互场景
- 边界测试：异常输入处理
"""

import asyncio
import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


# =============================================================================
# Mock API 应用（用于测试）
# =============================================================================


def create_test_app() -> FastAPI:
    """
    创建测试用 FastAPI 应用

    模拟 Towngas Manus 的 API 结构
    """
    app = FastAPI(
        title="Towngas Manus API",
        version="0.1.0-test",
        docs_url="/docs",
    )

    # 模拟数据存储
    sessions_db: Dict[str, Dict[str, Any]] = {}
    messages_db: Dict[str, list] = {}
    skills_db: Dict[str, Dict[str, Any]] = {
        "gas_price_query": {
            "name": "gas_price_query",
            "description": "查询燃气价格",
            "category": "inquiry",
        },
        "bill_calculator": {
            "name": "bill_calculator",
            "description": "计算燃气费用",
            "category": "utility",
        },
    }

    # =============================================================================
    # 健康检查端点
    # =============================================================================

    @app.get("/api/health")
    async def health_check():
        """健康检查端点"""
        return {
            "status": "healthy",
            "version": "0.1.0-test",
            "timestamp": "2024-01-15T10:00:00Z",
        }

    @app.get("/api/ready")
    async def readiness_check():
        """就绪检查端点"""
        return {
            "ready": True,
            "services": {
                "database": "connected",
                "cache": "connected",
                "llm": "available",
            },
        }

    # =============================================================================
    # 聊天 API
    # =============================================================================

    @app.post("/api/v1/chat")
    async def chat(request: Request):
        """
        聊天端点

        处理用户消息并返回 AI 响应
        """
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="无效的 JSON 格式")

        message = body.get("message")
        session_id = body.get("session_id")
        context = body.get("context", {})

        # 验证必需字段
        if not message:
            raise HTTPException(status_code=400, detail="消息不能为空")

        if not message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")

        # 自动生成会话 ID
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        # 创建或获取会话
        if session_id not in sessions_db:
            sessions_db[session_id] = {
                "session_id": session_id,
                "created_at": "2024-01-15T10:00:00Z",
                "status": "active",
            }
            messages_db[session_id] = []

        # 存储用户消息
        messages_db[session_id].append(
            {"role": "user", "content": message, "timestamp": "2024-01-15T10:00:00Z"}
        )

        # 模拟 AI 响应
        response_content = f"收到您的消息：{message}"

        # 检查是否需要调用技能
        skill_called = None
        if "价格" in message or "查询" in message:
            skill_called = "gas_price_query"
            response_content = f"已为您查询价格信息。"

        # 存储助手消息
        messages_db[session_id].append(
            {
                "role": "assistant",
                "content": response_content,
                "timestamp": "2024-01-15T10:00:01Z",
            }
        )

        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "response": response_content,
                "skill_called": skill_called,
                "model": "claude-3-sonnet",
            },
        }

    @app.post("/api/v1/chat/stream")
    async def chat_stream(request: Request):
        """
        流式聊天端点

        返回 Server-Sent Events 格式的流式响应
        """
        from fastapi.responses import StreamingResponse
        import json

        body = await request.json()
        message = body.get("message", "")

        async def generate():
            """生成流式响应"""
            response = f"收到您的消息：{message}"
            for i, char in enumerate(response):
                data = json.dumps({"chunk": char, "index": i})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    # =============================================================================
    # 会话 API
    # =============================================================================

    @app.post("/api/v1/sessions")
    async def create_session(request: Request):
        """创建新会话"""
        import uuid

        body = await request.json() if await request.body() else {}
        metadata = body.get("metadata", {}) if body else {}

        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:00:00Z",
            "status": "active",
            "metadata": metadata,
            "message_count": 0,
        }

        sessions_db[session_id] = session
        messages_db[session_id] = []

        return {"success": True, "data": session}

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(session_id: str):
        """获取会话信息"""
        if session_id not in sessions_db:
            raise HTTPException(status_code=404, detail="会话不存在")

        return {"success": True, "data": sessions_db[session_id]}

    @app.get("/api/v1/sessions")
    async def list_sessions(status: Optional[str] = None, limit: int = 100):
        """列出所有会话"""
        sessions = list(sessions_db.values())

        if status:
            sessions = [s for s in sessions if s.get("status") == status]

        return {"success": True, "data": sessions[:limit], "total": len(sessions)}

    @app.delete("/api/v1/sessions/{session_id}")
    async def delete_session(session_id: str):
        """删除会话"""
        if session_id not in sessions_db:
            raise HTTPException(status_code=404, detail="会话不存在")

        del sessions_db[session_id]
        if session_id in messages_db:
            del messages_db[session_id]

        return {"success": True, "message": "会话已删除"}

    @app.get("/api/v1/sessions/{session_id}/history")
    async def get_session_history(
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ):
        """获取会话历史"""
        if session_id not in sessions_db:
            raise HTTPException(status_code=404, detail="会话不存在")

        history = messages_db.get(session_id, [])

        start = min(offset, len(history))
        end = len(history) if limit is None else min(start + limit, len(history))

        return {
            "success": True,
            "data": history[start:end],
            "total": len(history),
        }

    # =============================================================================
    # 技能 API
    # =============================================================================

    @app.get("/api/v1/skills")
    async def list_skills(category: Optional[str] = None):
        """列出所有技能"""
        skills = list(skills_db.values())

        if category:
            skills = [s for s in skills if s.get("category") == category]

        return {"success": True, "data": skills, "total": len(skills)}

    @app.get("/api/v1/skills/{skill_name}")
    async def get_skill(skill_name: str):
        """获取技能详情"""
        if skill_name not in skills_db:
            raise HTTPException(status_code=404, detail="技能不存在")

        return {"success": True, "data": skills_db[skill_name]}

    @app.post("/api/v1/skills/{skill_name}/execute")
    async def execute_skill(skill_name: str, request: Request):
        """执行技能"""
        if skill_name not in skills_db:
            raise HTTPException(status_code=404, detail="技能不存在")

        body = await request.json()
        params = body.get("params", {})

        # 模拟技能执行
        if skill_name == "gas_price_query":
            result = {
                "region": params.get("region", "香港"),
                "price": 3.5,
                "unit": "HKD/m³",
            }
        elif skill_name == "bill_calculator":
            result = {
                "total": 350.0,
                "usage": params.get("usage", 100),
                "currency": "HKD",
            }
        else:
            result = {"executed": True}

        return {
            "success": True,
            "data": {
                "skill": skill_name,
                "result": result,
                "params": params,
            },
        }

    # =============================================================================
    # 错误处理
    # =============================================================================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP 异常处理"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                },
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """通用异常处理"""
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": 500,
                    "message": "内部服务器错误",
                },
            },
        )

    return app


# =============================================================================
# 测试类
# =============================================================================


class TestHealthEndpoints:
    """健康检查端点测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_test_app()
        return TestClient(app)

    @pytest.mark.integration
    def test_health_check_success(self, client):
        """测试健康检查成功"""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.integration
    def test_readiness_check_success(self, client):
        """测试就绪检查成功"""
        response = client.get("/api/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert "services" in data


class TestChatAPI:
    """聊天 API 测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_test_app()
        return TestClient(app)

    @pytest.mark.integration
    def test_chat_simple_message(self, client):
        """测试简单消息聊天"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "你好"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "response" in data["data"]
        assert "session_id" in data["data"]

    @pytest.mark.integration
    def test_chat_with_session_id(self, client):
        """测试带会话 ID 的聊天"""
        session_id = "test-session-123"
        response = client.post(
            "/api/v1/chat",
            json={"message": "你好", "session_id": session_id},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["session_id"] == session_id

    @pytest.mark.integration
    def test_chat_with_context(self, client):
        """测试带上下文的聊天"""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "你好",
                "context": {"user_id": "user-001", "language": "zh"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.integration
    def test_chat_empty_message_raises_error(self, client):
        """测试空消息抛出错误"""
        response = client.post(
            "/api/v1/chat",
            json={"message": ""},
        )

        assert response.status_code == 400

    @pytest.mark.integration
    def test_chat_whitespace_message_raises_error(self, client):
        """测试纯空白消息抛出错误"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "   \n\t  "},
        )

        assert response.status_code == 400

    @pytest.mark.integration
    def test_chat_invalid_json_raises_error(self, client):
        """测试无效 JSON 抛出错误"""
        response = client.post(
            "/api/v1/chat",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400

    @pytest.mark.integration
    def test_chat_triggers_skill(self, client):
        """测试消息触发技能"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "查询价格"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["skill_called"] == "gas_price_query"

    @pytest.mark.integration
    def test_chat_stream_endpoint_exists(self, client):
        """测试流式聊天端点存在"""
        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "测试流式"},
        )

        # 应该返回 200，即使我们不在测试中验证流式内容
        assert response.status_code == 200


class TestSessionAPI:
    """会话 API 测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_test_app()
        return TestClient(app)

    @pytest.mark.integration
    def test_create_session(self, client):
        """测试创建会话"""
        response = client.post(
            "/api/v1/sessions",
            json={"metadata": {"user_id": "user-001"}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "session_id" in data["data"]
        assert data["data"]["status"] == "active"

    @pytest.mark.integration
    def test_create_session_empty_body(self, client):
        """测试空请求体创建会话"""
        response = client.post(
            "/api/v1/sessions",
            json={},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.integration
    def test_get_session(self, client):
        """测试获取会话"""
        # 先创建会话
        create_response = client.post("/api/v1/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]

        # 获取会话
        response = client.get(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["session_id"] == session_id

    @pytest.mark.integration
    def test_get_non_existent_session(self, client):
        """测试获取不存在的会话"""
        response = client.get("/api/v1/sessions/non-existent")

        assert response.status_code == 404

    @pytest.mark.integration
    def test_list_sessions(self, client):
        """测试列出会话"""
        # 创建几个会话
        for _ in range(3):
            client.post("/api/v1/sessions", json={})

        response = client.get("/api/v1/sessions")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 3

    @pytest.mark.integration
    def test_list_sessions_with_limit(self, client):
        """测试限制会话列表数量"""
        # 创建几个会话
        for _ in range(5):
            client.post("/api/v1/sessions", json={})

        response = client.get("/api/v1/sessions?limit=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) <= 2

    @pytest.mark.integration
    def test_delete_session(self, client):
        """测试删除会话"""
        # 创建会话
        create_response = client.post("/api/v1/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]

        # 删除会话
        response = client.delete(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 200
        assert response.json()["success"] is True

        # 验证会话已删除
        get_response = client.get(f"/api/v1/sessions/{session_id}")
        assert get_response.status_code == 404

    @pytest.mark.integration
    def test_delete_non_existent_session(self, client):
        """测试删除不存在的会话"""
        response = client.delete("/api/v1/sessions/non-existent")

        assert response.status_code == 404

    @pytest.mark.integration
    def test_get_session_history(self, client):
        """测试获取会话历史"""
        # 创建会话并发送消息
        chat_response = client.post(
            "/api/v1/chat",
            json={"message": "测试消息"},
        )
        session_id = chat_response.json()["data"]["session_id"]

        # 获取历史
        response = client.get(f"/api/v1/sessions/{session_id}/history")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 2  # 用户消息 + 助手响应

    @pytest.mark.integration
    def test_get_non_existent_session_history(self, client):
        """测试获取不存在会话的历史"""
        response = client.get("/api/v1/sessions/non-existent/history")

        assert response.status_code == 404


class TestSkillsAPI:
    """技能 API 测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_test_app()
        return TestClient(app)

    @pytest.mark.integration
    def test_list_skills(self, client):
        """测试列出技能"""
        response = client.get("/api/v1/skills")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 2

    @pytest.mark.integration
    def test_list_skills_by_category(self, client):
        """测试按分类列出技能"""
        response = client.get("/api/v1/skills?category=inquiry")

        assert response.status_code == 200
        data = response.json()
        for skill in data["data"]:
            assert skill["category"] == "inquiry"

    @pytest.mark.integration
    def test_get_skill(self, client):
        """测试获取技能详情"""
        response = client.get("/api/v1/skills/gas_price_query")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["name"] == "gas_price_query"

    @pytest.mark.integration
    def test_get_non_existent_skill(self, client):
        """测试获取不存在的技能"""
        response = client.get("/api/v1/skills/non_existent")

        assert response.status_code == 404

    @pytest.mark.integration
    def test_execute_skill(self, client):
        """测试执行技能"""
        response = client.post(
            "/api/v1/skills/gas_price_query/execute",
            json={"params": {"region": "香港"}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data["data"]

    @pytest.mark.integration
    def test_execute_non_existent_skill(self, client):
        """测试执行不存在的技能"""
        response = client.post(
            "/api/v1/skills/non_existent/execute",
            json={"params": {}},
        )

        assert response.status_code == 404

    @pytest.mark.integration
    def test_execute_bill_calculator(self, client):
        """测试账单计算器技能"""
        response = client.post(
            "/api/v1/skills/bill_calculator/execute",
            json={"params": {"usage": 100}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["result"]["usage"] == 100


class TestAPIErrorHandling:
    """API 错误处理测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_test_app()
        return TestClient(app)

    @pytest.mark.integration
    def test_404_error_format(self, client):
        """测试 404 错误格式"""
        response = client.get("/api/v1/non-existent")

        assert response.status_code == 404

    @pytest.mark.integration
    def test_400_error_format(self, client):
        """测试 400 错误格式"""
        response = client.post(
            "/api/v1/chat",
            json={"message": ""},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data


class TestAPIIntegration:
    """API 集成测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_test_app()
        return TestClient(app)

    @pytest.mark.e2e
    def test_complete_chat_flow(self, client):
        """测试完整聊天流程"""
        # 1. 创建会话
        create_response = client.post("/api/v1/sessions", json={})
        assert create_response.status_code == 200
        session_id = create_response.json()["data"]["session_id"]

        # 2. 发送消息
        chat_response = client.post(
            "/api/v1/chat",
            json={"message": "你好", "session_id": session_id},
        )
        assert chat_response.status_code == 200

        # 3. 获取历史
        history_response = client.get(f"/api/v1/sessions/{session_id}/history")
        assert history_response.status_code == 200
        assert len(history_response.json()["data"]) >= 2

        # 4. 删除会话
        delete_response = client.delete(f"/api/v1/sessions/{session_id}")
        assert delete_response.status_code == 200

    @pytest.mark.e2e
    def test_skill_execution_flow(self, client):
        """测试技能执行流程"""
        # 1. 列出技能
        list_response = client.get("/api/v1/skills")
        assert list_response.status_code == 200
        skills = list_response.json()["data"]
        assert len(skills) > 0

        # 2. 获取技能详情
        skill_name = skills[0]["name"]
        detail_response = client.get(f"/api/v1/skills/{skill_name}")
        assert detail_response.status_code == 200

        # 3. 执行技能
        execute_response = client.post(
            f"/api/v1/skills/{skill_name}/execute",
            json={"params": {}},
        )
        assert execute_response.status_code == 200

    @pytest.mark.e2e
    def test_chat_triggers_skill_flow(self, client):
        """测试聊天触发技能流程"""
        # 发送触发技能的消息
        response = client.post(
            "/api/v1/chat",
            json={"message": "我想查询价格"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["skill_called"] is not None


class TestAPIEdgeCases:
    """API 边界条件测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app = create_test_app()
        return TestClient(app)

    @pytest.mark.integration
    def test_chat_with_unicode(self, client):
        """测试 Unicode 消息"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "你好世界🎉🎊中文测试"},
        )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_chat_with_long_message(self, client):
        """测试长消息"""
        long_message = "这是一段很长的消息。" * 1000
        response = client.post(
            "/api/v1/chat",
            json={"message": long_message},
        )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_chat_with_special_characters(self, client):
        """测试特殊字符消息"""
        special_chars = "!@#$%^&*(){}[]|\\:;\"'<>,.?/~`\n\t\r"
        response = client.post(
            "/api/v1/chat",
            json={"message": special_chars},
        )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_session_with_large_metadata(self, client):
        """测试大量元数据的会话"""
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(50)}

        response = client.post(
            "/api/v1/sessions",
            json={"metadata": large_metadata},
        )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_concurrent_chat_requests(self, client):
        """测试并发聊天请求"""
        import concurrent.futures

        def send_chat(i):
            return client.post(
                "/api/v1/chat",
                json={"message": f"消息 {i}"},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_chat, i) for i in range(10)]
            results = [f.result() for f in futures]

        # 所有请求都应该成功
        assert all(r.status_code == 200 for r in results)
