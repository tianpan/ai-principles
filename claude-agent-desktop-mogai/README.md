# Towngas Manus - 港华智能体平台

<p align="center">
  <strong>企业级自主 Agent 平台 | 基于 Claude Agent SDK 构建</strong>
</p>

<p align="center">
  <a href="#项目介绍">项目介绍</a> •
  <a href="#技术栈">技术栈</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#项目结构">项目结构</a> •
  <a href="#api-文档">API 文档</a> •
  <a href="#开发指南">开发指南</a> •
  <a href="#部署说明">部署说明</a>
</p>

---

## 项目介绍

**Towngas Manus** 是港华集团的企业级自主 Agent 平台，基于 Anthropic Claude Agent SDK 构建。

### 核心特性

- **强自主性** - 不只是问答，而是能规划、执行、反馈的智能体
- **统一平台** - 一个平台服务 TOP、工程移动等所有自研系统
- **企业级** - 面向港华内部用户，安全可控
- **可扩展** - 支持自定义 Skills 和 Subagents

### 架构理念

采用"**1 主 Agent + N Skills + M Subagents**"模式：

```
┌─────────────────────────────────────────────────┐
│              Manus Agent (主控)                  │
│  理解用户意图 → 路由到对应 Skill/Subagent        │
└───────────────────────┬─────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Skills    │ │   Skills    │ │  Subagents  │
│  (MCP工具)  │ │  (MCP工具)  │ │ (复杂任务)   │
│  场站查询   │ │  知识问答   │ │ 日报生成    │
│  设备诊断   │ │  气量预测   │ │ 应急处置    │
└─────────────┘ └─────────────┘ └─────────────┘
```

### 覆盖范围

| 系统 | 说明 |
|------|------|
| TOP | 场站管网业务 |
| 工程移动 | 工程场景 |
| 其他自研系统 | 未来扩展 |
| BIP | 边界外（由用友 YonAI 负责） |

---

## 技术栈

### 后端

| 技术 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 主要开发语言 |
| FastAPI | 0.109+ | 异步 Web 框架 |
| Anthropic SDK | 0.39+ | Claude API SDK |
| Pydantic | 2.6+ | 数据验证 |
| Uvicorn | 0.27+ | ASGI 服务器 |

### 前端

| 技术 | 版本 | 说明 |
|------|------|------|
| Vue.js | 3.4+ | 前端框架 |
| TypeScript | 5.4+ | 类型支持 |
| Vite | 5.2+ | 构建工具 |
| Axios | 1.6+ | HTTP 客户端 |
| Marked | 12.0+ | Markdown 解析 |

### 基础设施

| 技术 | 说明 |
|------|------|
| Docker | 容器化 |
| Docker Compose | 服务编排 |
| Redis | 缓存（可选） |

---

## 快速开始

### 前置要求

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose（生产环境）
- Anthropic API Key

### 方式一：Docker 部署（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/towngas/manus.git
cd manus

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 ANTHROPIC_API_KEY

# 3. 启动服务
chmod +x scripts/start.sh
./scripts/start.sh

# 4. 访问服务
# 前端界面: http://localhost:8000
# API 文档: http://localhost:8000/docs
```

### 方式二：本地开发

```bash
# 1. 克隆项目
git clone https://github.com/towngas/manus.git
cd manus

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 3. 启动开发环境（自动安装依赖）
chmod +x scripts/dev.sh
./scripts/dev.sh --install

# 4. 访问服务
# 前端界面: http://localhost:5173
# 后端 API: http://localhost:8000
```

### 环境变量说明

| 变量 | 必填 | 说明 |
|------|------|------|
| `ANTHROPIC_API_KEY` | 是 | Anthropic API 密钥 |
| `ANTHROPIC_MODEL` | 否 | 使用的模型（默认 claude-sonnet-4-5） |
| `LOG_LEVEL` | 否 | 日志级别（默认 INFO） |
| `CORS_ORIGINS` | 否 | CORS 允许的源（默认 *） |

---

## 项目结构

```
towngas-manus/
├── backend/                    # 后端代码
│   ├── main.py                # 应用入口
│   ├── config.py              # 配置管理
│   ├── agents/                # Agent 实现
│   │   ├── manus.py          # 主 Agent
│   │   └── subagents/        # 子代理
│   ├── skills/                # 技能模块
│   │   ├── registry.py       # 技能注册表
│   │   └── towngas/          # 港华专属技能
│   ├── adapters/              # 数据适配器
│   │   ├── top/              # TOP 系统适配器
│   │   └── knowledge/        # 知识库适配器
│   ├── api/                   # API 路由
│   │   ├── chat.py           # 聊天接口
│   │   └── sessions.py       # 会话管理
│   └── requirements.txt       # Python 依赖
│
├── frontend/                   # 前端代码
│   ├── src/
│   │   ├── App.vue           # 主组件
│   │   ├── components/       # UI 组件
│   │   ├── api/              # API 调用
│   │   └── utils/            # 工具函数
│   ├── package.json          # Node 依赖
│   └── vite.config.ts        # Vite 配置
│
├── docs/                       # 文档
│   └── PRD-Towngas-Manus.md   # 产品需求文档
│
├── scripts/                    # 脚本
│   ├── start.sh               # 生产启动脚本
│   └── dev.sh                 # 开发启动脚本
│
├── data/                       # 数据目录
├── logs/                       # 日志目录
├── files/                      # 文件存储
│
├── Dockerfile                  # Docker 构建文件
├── docker-compose.yml          # Docker Compose 配置
├── .env.example               # 环境变量示例
├── .gitignore                 # Git 忽略配置
└── README.md                  # 本文件
```

---

## API 文档

### 基础接口

#### 健康检查

```http
GET /health
```

响应：
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

#### 聊天接口

```http
POST /api/chat
Content-Type: application/json

{
  "message": "查询场站 A001 的运行状态",
  "session_id": "session-123"
}
```

响应（SSE 流式）：
```
data: {"type": "text", "content": "正在查询场站 A001..."}
data: {"type": "text", "content": "场站 A001 运行正常"}
data: {"type": "done"}
```

#### 会话管理

```http
# 创建会话
POST /api/sessions

# 获取会话列表
GET /api/sessions

# 获取会话历史
GET /api/sessions/{session_id}/messages

# 删除会话
DELETE /api/sessions/{session_id}
```

### Skills 接口

```http
# 获取可用技能列表
GET /api/skills

# 获取技能详情
GET /api/skills/{skill_name}
```

### 完整 API 文档

启动服务后访问：http://localhost:8000/docs

---

## 开发指南

### 开发环境配置

1. **安装后端依赖**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

2. **安装前端依赖**

```bash
cd frontend
npm install
cd ..
```

3. **启动开发服务**

```bash
# 后端
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# 前端（新终端）
cd frontend
npm run dev
```

### 开发规范

#### 代码规范

- Python: 遵循 PEP 8，使用 Black 格式化
- TypeScript: 使用 ESLint + Prettier
- 提交信息: 遵循 Conventional Commits

#### 分支管理

```
main        # 主分支，稳定版本
develop     # 开发分支
feature/*   # 功能分支
bugfix/*    # 修复分支
release/*   # 发布分支
```

#### 提交规范

```
feat: 添加新功能
fix: 修复 bug
docs: 文档更新
style: 代码格式
refactor: 重构
test: 测试相关
chore: 构建/工具
```

### 添加新 Skill

1. 在 `backend/skills/towngas/` 创建技能文件：

```python
# backend/skills/towngas/my_skill.py
from backend.skills.base import Skill

class MySkill(Skill):
    name = "my_skill"
    description = "技能描述"

    async def execute(self, **params):
        # 实现技能逻辑
        return {"result": "success"}
```

2. 在 `backend/skills/registry.py` 注册：

```python
from backend.skills.towngas.my_skill import MySkill

SKILLS = {
    "my_skill": MySkill(),
}
```

### 添加新 Subagent

参考 `backend/agents/subagents/` 目录下的实现。

---

## 部署说明

### Docker 部署（推荐）

#### 单机部署

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

#### 生产环境配置

1. **修改 docker-compose.yml**

```yaml
services:
  backend:
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "2"
          memory: 4G
```

2. **配置 HTTPS**

使用 Nginx 反向代理：

```nginx
server {
    listen 443 ssl;
    server_name manus.towngas.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
    }
}
```

### Kubernetes 部署

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: towngas-manus
spec:
  replicas: 3
  selector:
    matchLabels:
      app: towngas-manus
  template:
    spec:
      containers:
      - name: backend
        image: towngas/manus:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: manus-secrets
              key: anthropic-api-key
```

### 环境变量配置

| 变量 | 说明 | 生产环境建议 |
|------|------|-------------|
| `LOG_LEVEL` | 日志级别 | WARNING 或 ERROR |
| `CORS_ORIGINS` | CORS 源 | 具体域名列表 |
| `DATABASE_URL` | 数据库连接 | PostgreSQL |
| `REDIS_URL` | Redis 连接 | 用于缓存和会话 |

---

## 版本规划

| 版本 | 状态 | 内容 |
|------|------|------|
| v0.1 MVP | 已完成 | 核心功能、基础架构 |
| v0.2 数据接入 | 进行中 | TOP API、知识库、港华 Skills |
| v0.3 增强 | 计划中 | 企业认证、消息推送 |
| v1.0 Agent OS | 计划中 | 多租户、权限、审计 |

---

## 常见问题

### Q: 如何获取 Anthropic API Key？

访问 [Anthropic Console](https://console.anthropic.com/) 注册并创建 API Key。

### Q: 支持哪些 Claude 模型？

- claude-sonnet-4-5（推荐，平衡性能与成本）
- claude-opus-4-5（最强推理）
- claude-haiku-4-5（最快响应）

### Q: 如何连接 TOP 系统？

在 v0.2 版本中，通过 TOP Adapter 实现。需要配置：

```env
TOP_API_URL=https://top.towngas.com/api
TOP_API_KEY=your-top-api-key
```

### Q: 如何添加知识库？

1. 准备文档（PDF、Word、Markdown）
2. 使用向量化工具处理
3. 配置知识库连接

---

## 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: 添加某功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 许可证

本项目为港华集团内部项目，未经授权不得对外分发。

---

## 联系方式

- 项目负责人：田攀
- 技术支持：towngas-it@towngas.com

---

<p align="center">
  <strong>Towngas Manus</strong> - 港华的智能执行者
</p>
