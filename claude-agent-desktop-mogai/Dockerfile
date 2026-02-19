# ===========================================
# Towngas Manus - 多阶段 Docker 构建文件
# ===========================================
# 项目：港华智能体平台
# 技术栈：Python FastAPI 后端 + Vue3 前端
# ===========================================

# ===========================================
# 阶段1: 前端构建
# ===========================================
FROM node:20-alpine AS frontend-builder

# 设置工作目录
WORKDIR /app/frontend

# 设置 npm 镜像源（国内加速）
RUN npm config set registry https://registry.npmmirror.com

# 复制前端依赖文件
COPY frontend/package.json ./
COPY frontend/package-lock.json* ./

# 安装依赖
RUN npm ci --silent || npm install --silent

# 复制前端源码
COPY frontend/ ./

# 构建前端（生产环境）
RUN npm run build

# ===========================================
# 阶段2: 后端运行时
# ===========================================
FROM python:3.11-slim AS backend

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # 应用配置
    APP_NAME="Towngas Manus" \
    APP_VERSION="1.0.0" \
    # 服务配置
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    # 日志级别
    LOG_LEVEL=INFO

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 网络工具
    curl \
    # 编译依赖（某些 Python 包需要）
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户（安全最佳实践）
RUN groupadd -r towngas && useradd -r -g towngas towngas

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY backend/requirements.txt ./requirements.txt

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端源码
COPY backend/ ./backend/

# 从前端构建阶段复制静态文件
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# 创建必要的目录
RUN mkdir -p /app/data /app/logs /app/files && \
    chown -R towngas:towngas /app

# 切换到非 root 用户
USER towngas

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
