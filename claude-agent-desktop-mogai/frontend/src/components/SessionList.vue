<!--
  Towngas Manus Frontend - 会话列表组件

  功能：
  1. 显示会话列表
  2. 创建新会话
  3. 切换会话
  4. 删除会话
-->

<template>
  <div class="session-list">
    <!-- 头部：标题和新建按钮 -->
    <div class="session-header">
      <h2 class="session-title">会话列表</h2>
      <button
        class="new-session-btn"
        @click="handleCreate"
        title="创建新会话 (Cmd+N)"
      >
        <Plus :size="16" />
      </button>
    </div>

    <!-- 搜索框 -->
    <div class="search-box">
      <Search class="search-icon" :size="16" />
      <input
        v-model="searchQuery"
        type="text"
        class="search-input"
        placeholder="搜索会话..."
      />
    </div>

    <!-- 会话列表 -->
    <div class="session-items">
      <!-- 空状态 -->
      <div v-if="filteredSessions.length === 0" class="empty-sessions">
        <MessageSquare class="empty-icon" :size="32" />
        <p v-if="searchQuery">未找到匹配的会话</p>
        <p v-else>暂无会话，点击上方按钮创建</p>
      </div>

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
          <MessageCircle :size="18" />
        </div>

        <!-- 会话信息 -->
        <div class="session-info">
          <div class="session-name">{{ session.title }}</div>
          <div class="session-meta">
            <span class="session-time">{{ formatDate(session.updatedAt) }}</span>
            <span v-if="session.messageCount" class="session-count">
              {{ session.messageCount }} 条消息
            </span>
          </div>
        </div>

        <!-- 删除按钮 -->
        <button
          class="delete-btn"
          @click.stop="handleDelete(session.id)"
          title="删除会话"
        >
          <Trash2 :size="14" />
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 会话列表组件逻辑
 */

import { ref, computed } from 'vue'
import { Plus, Search, MessageCircle, MessageSquare, Trash2 } from 'lucide-vue-next'
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

/* 新建会话按钮 */
.new-session-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 8px;
  background-color: var(--primary-color);
  color: white;
  cursor: pointer;
  transition: background-color 0.2s, transform 0.1s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.new-session-btn:hover {
  background-color: var(--primary-color-dark);
  transform: scale(1.05);
}

/* 搜索框 */
.search-box {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  gap: 8px;
}

.search-icon {
  color: var(--text-muted);
  flex-shrink: 0;
}

.search-input {
  flex: 1;
  padding: 8px 0;
  border: none;
  background-color: transparent;
  color: var(--text-primary);
  font-size: 0.875rem;
  outline: none;
}

.search-input::placeholder {
  color: var(--text-muted);
}

/* 会话列表区域 */
.session-items {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

/* 空状态 */
.empty-sessions {
  padding: 40px 16px;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.875rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.empty-icon {
  opacity: 0.5;
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
  gap: 8px;
  margin-top: 4px;
  font-size: 0.75rem;
  color: var(--text-muted);
}

/* 删除按钮 */
.delete-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 6px;
  background-color: transparent;
  color: var(--text-muted);
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s, background-color 0.2s, color 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.session-item:hover .delete-btn {
  opacity: 1;
}

.delete-btn:hover {
  background-color: var(--danger-color-alpha);
  color: var(--danger-color);
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
