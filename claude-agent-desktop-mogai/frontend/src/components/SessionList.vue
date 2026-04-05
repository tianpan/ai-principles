<!--
  Towngas Manus Frontend - 会话列表组件

  功能：
  1. 显示会话列表
  2. 创建新会话
  3. 切换会话
  4. 删除会话

  已迁移到 Element Plus：
  - el-button type="primary" circle 替代新建按钮
  - el-input :prefix-icon="Search" 替代搜索框
  - el-button type="danger" text circle 替代删除按钮
  - el-tag 替代消息数标签
  - el-empty 替代空状态
-->

<template>
  <div class="session-list">
    <!-- 头部：标题和新建按钮 -->
    <div class="session-header">
      <h2 class="session-title">会话列表</h2>
      <el-button
        type="primary"
        circle
        @click="handleCreate"
        title="创建新会话 (Cmd+N)"
      >
        <el-icon><Plus /></el-icon>
      </el-button>
    </div>

    <!-- Element Plus 搜索框 -->
    <div class="search-box">
      <el-input
        v-model="searchQuery"
        :prefix-icon="Search"
        placeholder="搜索会话..."
        clearable
      />
    </div>

    <!-- 会话列表 -->
    <div class="session-items">
      <!-- Element Plus 空状态 -->
      <el-empty
        v-if="filteredSessions.length === 0"
        :image-size="80"
        :description="searchQuery ? '未找到匹配的会话' : '暂无会话，点击上方按钮创建'"
      >
        <template #image>
          <el-icon :size="48" color="var(--text-muted)">
            <ChatDotRound />
          </el-icon>
        </template>
      </el-empty>

      <!-- 会话项 -->
      <div
        v-for="session in filteredSessions"
        :key="session.id"
        :class="[
          'session-item',
          { active: session.id === currentSessionId },
        ]"
        @click="handleSelect(session.id)"
      >
        <!-- 会话图标 -->
        <div class="session-icon">
          <el-icon :size="18"><ChatLineRound /></el-icon>
        </div>

        <!-- 会话信息 -->
        <div class="session-info">
          <div class="session-name">{{ session.title }}</div>
          <div class="session-meta">
            <span class="session-time">{{ formatDate(session.updatedAt) }}</span>
            <el-tag
              v-if="session.messageCount"
              size="small"
              type="info"
              class="session-count"
            >
              {{ session.messageCount }} 条消息
            </el-tag>
          </div>
        </div>

        <!-- Element Plus 删除按钮 -->
        <el-button
          type="danger"
          text
          circle
          class="delete-btn"
          @click.stop="handleDelete(session.id)"
          title="删除会话"
        >
          <el-icon><Delete /></el-icon>
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 会话列表组件逻辑
 * 已迁移到 Element Plus 组件库
 */

import { ref, computed } from 'vue'
// Element Plus 图标
import { Plus, Search, ChatDotRound, ChatLineRound, Delete } from '@element-plus/icons-vue'
import type { Session } from '@/types'
import { useConfirm } from '@/composables/useConfirm'

// ==================== Props定义 ====================

interface Props {
  /** 会话列表 */
  sessions: Session[]
  /** 当前选中的会话ID */
  currentSessionId: string
}

const props = defineProps<Props>()

// ==================== Emits定义 ====================

const emit = defineEmits<{
  /** 选择会话事件 */
  (e: 'select', sessionId: string): void
  /** 创建会话事件 */
  (e: 'create'): void
  /** 删除会话事件 */
  (e: 'delete', sessionId: string): void
}>()

// ==================== 状态 ====================

// 搜索查询
const searchQuery = ref('')

// 确认对话框
const { show } = useConfirm()

// ==================== 计算属性 ====================

// 过滤后的会话列表
const filteredSessions = computed(() => {
  if (!searchQuery.value.trim()) {
    return props.sessions
  }

  const query = searchQuery.value.toLowerCase()
  return props.sessions.filter((session) =>
    session.title.toLowerCase().includes(query)
  )
})

// ==================== 方法 ====================

/**
 * 格式化日期
 *
 * @param dateString ISO格式日期字符串
 * @returns 格式化后的日期字符串
 */
const formatDate = (dateString: string): string => {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  // 今天
  if (diffDays === 0) {
    return date.toLocaleTimeString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  // 昨天
  if (diffDays === 1) {
    return '昨天'
  }

  // 一周内
  if (diffDays < 7) {
    return `${diffDays}天前`
  }

  // 更早
  return date.toLocaleDateString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
  })
}

/**
 * 处理会话选择
 *
 * @param sessionId 会话ID
 */
const handleSelect = (sessionId: string): void => {
  emit('select', sessionId)
}

/**
 * 处理创建新会话
 */
const handleCreate = (): void => {
  emit('create')
}

/**
 * 处理删除会话
 *
 * @param sessionId 会话ID
 */
const handleDelete = async (sessionId: string): Promise<void> => {
  // 使用确认对话框
  const confirmed = await show({
    title: '删除会话',
    message: '确定要删除这个会话吗？此操作不可撤销。',
    confirmText: '删除',
    cancelText: '取消',
    variant: 'danger',
  })

  if (confirmed) {
    emit('delete', sessionId)
  }
}
</script>

<style scoped>
/* 会话列表容器 */
.session-list {
  display: flex;
  flex-direction: column;
  height: 100%;
}

/* 头部 */
.session-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--border-color);
}

.session-title {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

/* 搜索框 - Element Plus 覆盖 */
.search-box {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-color);
}

.search-box :deep(.el-input__wrapper) {
  background-color: transparent;
  box-shadow: none;
  padding: 0;
}

.search-box :deep(.el-input__inner) {
  padding: 8px 0;
  font-size: 0.875rem;
}

/* 会话列表区域 */
.session-items {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

/* Element Plus 空状态 */
.session-items :deep(.el-empty) {
  padding: 40px 16px;
}

/* 会话项 */
.session-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.2s;
  position: relative;
}

.session-item:hover {
  background-color: var(--hover-bg);
}

.session-item.active {
  background-color: var(--primary-color-alpha);
}

/* 会话图标 */
.session-icon {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background-color: var(--bg-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary);
  flex-shrink: 0;
}

.session-item.active .session-icon {
  color: var(--primary-color);
  background-color: var(--primary-color-alpha);
}

/* 会话信息 */
.session-info {
  flex: 1;
  min-width: 0;
}

.session-name {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 4px;
  font-size: 0.75rem;
  color: var(--text-muted);
}

/* 消息数标签 */
.session-count {
  font-size: 0.625rem;
}

/* 删除按钮 */
.delete-btn {
  opacity: 0;
  transition: opacity 0.2s;
}

.session-item:hover .delete-btn {
  opacity: 1;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .session-header {
    padding: 12px;
  }

  .session-items {
    padding: 4px;
  }

  .session-item {
    padding: 10px;
  }
}
</style>
