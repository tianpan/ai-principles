"""
Towngas Manus 测试套件
======================

测试模块包含：
- test_agent_engine.py: Agent Engine 测试
- test_session_manager.py: Session Manager 测试
- test_skills_registry.py: Skills Registry 测试
- test_api.py: API 集成测试

运行测试：
    pytest backend/tests/

运行覆盖率报告：
    pytest --cov=backend/app --cov-report=html

测试标记：
    @pytest.mark.unit - 单元测试
    @pytest.mark.integration - 集成测试
    @pytest.mark.e2e - 端到端测试
    @pytest.mark.slow - 慢速测试
"""

__version__ = "0.1.0"
__author__ = "Towngas Manus Team"
