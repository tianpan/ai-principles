#!/bin/bash

# ===========================================
# Towngas Manus - 开发环境启动脚本
# ===========================================
# 用途：启动开发环境（前后端分离，支持热重载）
# 使用：./scripts/dev.sh
# ===========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印横幅
print_banner() {
    echo ""
    echo -e "${PURPLE}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                                           ║${NC}"
    echo -e "${PURPLE}║        Towngas Manus 开发环境            ║${NC}"
    echo -e "${PURPLE}║        Development Mode                   ║${NC}"
    echo -e "${PURPLE}║                                           ║${NC}"
    echo -e "${PURPLE}╚═══════════════════════════════════════════╝${NC}"
    echo ""
}

# 检查环境变量
check_env() {
    log_info "检查环境变量..."

    if [ ! -f ".env" ]; then
        log_warning ".env 文件不存在，正在从 .env.example 复制..."
        cp .env.example .env
        log_warning "请编辑 .env 文件并填入实际的配置值"
    fi

    # 加载环境变量
    export $(grep -v '^#' .env | xargs 2>/dev/null || true)

    # 检查必要的环境变量
    if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-api-key-here" ]; then
        log_warning "ANTHROPIC_API_KEY 未设置，部分功能可能无法使用"
    fi

    log_success "环境变量检查完成"
}

# 创建必要目录
create_directories() {
    mkdir -p data logs files
}

# 检查 Python
check_python() {
    log_info "检查 Python 环境..."

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 未安装"
        exit 1
    fi

    # 检查 Python 版本
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_success "Python 版本: $PYTHON_VERSION"
}

# 检查 Node.js
check_node() {
    log_info "检查 Node.js 环境..."

    if ! command -v node &> /dev/null; then
        log_error "Node.js 未安装"
        exit 1
    fi

    NODE_VERSION=$(node --version)
    log_success "Node.js 版本: $NODE_VERSION"
}

# 安装后端依赖
install_backend_deps() {
    log_info "安装后端依赖..."

    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        log_info "创建虚拟环境..."
        python3 -m venv venv
        source venv/bin/activate
    fi

    pip install -r backend/requirements.txt

    log_success "后端依赖安装完成"
}

# 安装前端依赖
install_frontend_deps() {
    log_info "安装前端依赖..."

    cd frontend

    if [ -f "package-lock.json" ]; then
        npm ci
    else
        npm install
    fi

    cd ..

    log_success "前端依赖安装完成"
}

# 启动后端服务
start_backend() {
    log_info "启动后端服务..."

    # 在后台启动后端
    source venv/bin/activate

    # 设置开发环境变量
    export RELOAD=true
    export LOG_LEVEL=DEBUG

    # 启动 uvicorn（支持热重载）
    nohup python -m uvicorn backend.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --reload-dir backend \
        > logs/backend.log 2>&1 &

    BACKEND_PID=$!
    echo $BACKEND_PID > /tmp/towngas_backend.pid

    log_success "后端服务已启动 (PID: $BACKEND_PID)"
}

# 启动前端开发服务器
start_frontend() {
    log_info "启动前端开发服务器..."

    cd frontend

    # 在后台启动前端
    nohup npm run dev > ../logs/frontend.log 2>&1 &

    FRONTEND_PID=$!
    echo $FRONTEND_PID > /tmp/towngas_frontend.pid

    cd ..

    log_success "前端服务已启动 (PID: $FRONTEND_PID)"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."

    # 等待后端
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            break
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo ""
    log_success "服务已就绪"
}

# 显示服务信息
show_info() {
    echo ""
    log_info "开发环境已启动！"
    echo ""
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}  访问地址${NC}"
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo ""
    echo "  前端界面:    http://localhost:5173"
    echo "  后端 API:    http://localhost:8000"
    echo "  API 文档:    http://localhost:8000/docs"
    echo "  健康检查:    http://localhost:8000/health"
    echo ""
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}  日志文件${NC}"
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo ""
    echo "  后端日志:    tail -f logs/backend.log"
    echo "  前端日志:    tail -f logs/frontend.log"
    echo ""
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}  停止服务${NC}"
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo ""
    echo "  ./scripts/stop.sh"
    echo "  或者手动: kill \$(cat /tmp/towngas_backend.pid)"
    echo ""
}

# 清理函数
cleanup() {
    log_info "正在停止服务..."

    if [ -f /tmp/towngas_backend.pid ]; then
        kill $(cat /tmp/towngas_backend.pid) 2>/dev/null || true
        rm /tmp/towngas_backend.pid
    fi

    if [ -f /tmp/towngas_frontend.pid ]; then
        kill $(cat /tmp/towngas_frontend.pid) 2>/dev/null || true
        rm /tmp/towngas_frontend.pid
    fi

    log_success "服务已停止"
    exit 0
}

# 注册信号处理
trap cleanup SIGINT SIGTERM

# 主函数
main() {
    print_banner

    check_env
    create_directories
    check_python
    check_node

    # 检查是否需要安装依赖
    if [ "$1" = "--install" ] || [ "$1" = "-i" ]; then
        install_backend_deps
        install_frontend_deps
    fi

    start_backend
    start_frontend
    wait_for_services
    show_info

    # 保持脚本运行
    log_info "按 Ctrl+C 停止服务..."
    wait
}

# 运行主函数
main "$@"
