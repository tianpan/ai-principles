#!/bin/bash

# ===========================================
# Towngas Manus - 生产环境启动脚本
# ===========================================
# 用途：启动生产环境服务
# 使用：./scripts/start.sh
# ===========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                           ║${NC}"
    echo -e "${GREEN}║        Towngas Manus v1.0.0              ║${NC}"
    echo -e "${GREEN}║        港华智能体平台                     ║${NC}"
    echo -e "${GREEN}║                                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
    echo ""
}

# 检查环境变量
check_env() {
    log_info "检查环境变量..."

    if [ ! -f ".env" ]; then
        log_warning ".env 文件不存在，正在从 .env.example 复制..."
        cp .env.example .env
        log_warning "请编辑 .env 文件并填入实际的配置值"
        exit 1
    fi

    # 加载环境变量
    export $(grep -v '^#' .env | xargs)

    # 检查必要的环境变量
    if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-api-key-here" ]; then
        log_error "请设置 ANTHROPIC_API_KEY 环境变量"
        exit 1
    fi

    log_success "环境变量检查通过"
}

# 创建必要目录
create_directories() {
    log_info "创建必要的目录..."

    mkdir -p data logs files

    log_success "目录创建完成"
}

# 检查 Docker
check_docker() {
    log_info "检查 Docker..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi

    log_success "Docker 检查通过"
}

# 构建镜像
build_image() {
    log_info "构建 Docker 镜像..."

    docker-compose build --no-cache

    log_success "镜像构建完成"
}

# 启动服务
start_services() {
    log_info "启动服务..."

    docker-compose up -d

    log_success "服务启动完成"
}

# 等待服务就绪
wait_for_service() {
    log_info "等待服务就绪..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "服务已就绪"
            return 0
        fi

        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo ""
    log_error "服务启动超时"
    return 1
}

# 显示服务状态
show_status() {
    echo ""
    log_info "服务状态："
    docker-compose ps
    echo ""

    log_info "访问地址："
    echo "  - 前端界面: http://localhost:8000"
    echo "  - API 文档: http://localhost:8000/docs"
    echo "  - 健康检查: http://localhost:8000/health"
    echo ""

    log_info "日志查看："
    echo "  docker-compose logs -f backend"
    echo ""

    log_info "停止服务："
    echo "  docker-compose down"
    echo ""
}

# 主函数
main() {
    print_banner

    check_env
    create_directories
    check_docker
    build_image
    start_services
    wait_for_service
    show_status
}

# 运行主函数
main "$@"
