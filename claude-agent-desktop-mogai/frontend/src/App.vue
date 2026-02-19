<!--
  Towngas Manus Frontend - 根组件

  这是应用的根组件，负责：
  1. 整体布局
  2. 管理会话状态
  3. 协调子组件
-->

<template>
  <div class="app-container">
    <!-- 顶部导航栏 -->
    <header class="app-header">
      <div class="app-brand">
        <Flame class="app-logo" :size="28" />
        <h1 class="app-title">Towngas Manus</h1>
      </div>
      <span class="app-subtitle">AI Agent Assistant</span>
      <div class="app-actions">
        <button
          class="header-btn"
          @click="createSession"
          title="新建会话 (Cmd+N)"
        >
          <Plus :size="18" />
        </button>
        <button
          class="header-btn"
          @click="toggleSidebar"
          title="切换侧边栏 (Cmd+\)"
        >
          <PanelLeft :size="18" />
        </button>
        <button
          class="header-btn"
          @click="showHelp"
          title="帮助 (Cmd+/)"
        >
          <HelpCircle :size="18" />
        </button>
      </div>
    </header>

    <!-- 主内容区域 -->
    <main class="app-main" :class="{ 'sidebar-collapsed': sidebarCollapsed }">
      <!-- 左侧边栏 - 会话列表 -->
      <aside class="sidebar" :class="{ collapsed: sidebarCollapsed }">
        <SessionList
          :sessions="sessions"
          :current-session-id="currentSessionId"
          @select="selectSession"
          @create="createSession"
          @delete="deleteSession"
        />
      </aside>

      <!-- 中间聊天区域 -->
      <section class="chat-area">
        <ChatInterface
          :messages="currentMessages"
          :is-loading="isLoading"
          @send="sendMessage"
        />
      </section>

      <!-- 右侧边栏 - 技能面板 -->
      <aside class="skill-panel">
        <SkillPanel
          :skills="skills"
          @execute="executeSkill"
          @refresh="refreshSkills"
        />
      </aside>
    </main>

    <!-- 全局组件 -->
    <ToastContainer />
    <GlobalDialog />
  </div>
</template>

<script setup lang="ts">
/**
 * 根组件逻辑
 *
 * 使用Vue3 Composition API管理应用状态
 */

import { ref, computed, onMounted } from 'vue'
import { Flame, Plus, PanelLeft, HelpCircle } from 'lucide-vue-next'
import ChatInterface from './components/ChatInterface.vue'
import SessionList from './components/SessionList.vue'
import SkillPanel from './components/SkillPanel.vue'
import ToastContainer from './components/ui/ToastContainer.vue'
import GlobalDialog from './components/ui/GlobalDialog.vue'
import type { Message, Session, Skill } from './types'
import { chatApi, sessionApi, skillApi } from './api/chat'
import { useToast } from './composables/useToast'
import { useKeyboard } from './composables/useKeyboard'

// ==================== Toast ====================

const { success, error, warning, info } = useToast()

// ==================== 状态定义 ====================

// 会话列表
const sessions = ref<Session[]>([])
// 当前选中的会话ID
const currentSessionId = ref<string>('')
// 当前会话的消息列表
const messagesMap = ref<Map<string, Message[]>>(new Map())
// 加载状态
const isLoading = ref(false)
// 可用技能列表
const skills = ref<Skill[]>([])
// 侧边栏折叠状态
const sidebarCollapsed = ref(false)

/**
 * 切换侧边栏
 */
function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

/**
 * 显示帮助
 */
function showHelp() {
  info('快捷键：Cmd+N 新建会话，Cmd+\\ 切换侧边栏，Esc 关闭对话框')
}

// ==================== 键盘快捷键 ====================

useKeyboard([
  {
    key: 'n',
    ctrl: true,
    handler: () => createSession(),
    description: '新建会话',
  },
  {
    key: '\\',
    ctrl: true,
    handler: toggleSidebar,
    description: '切换侧边栏',
  },
  {
    key: '/',
    ctrl: true,
    handler: showHelp,
    description: '显示帮助',
  },
])

// ==================== 计算属性 ====================

// 获取当前会话的消息列表
const currentMessages = computed(() => {
  if (!currentSessionId.value) return []
  return messagesMap.value.get(currentSessionId.value) || []
})

// ==================== 会话管理方法 ====================

/**
 * 选择会话
 * @param sessionId 会话ID
 */
const selectSession = (sessionId: string) => {
  currentSessionId.value = sessionId
  // 如果该会话没有消息，则加载消息
  if (!messagesMap.value.has(sessionId)) {
    loadSessionMessages(sessionId)
  }
}

/**
 * 创建新会话
 */
const createSession = async () => {
  try {
    const newSession = await sessionApi.create()
    sessions.value.unshift(newSession)
    currentSessionId.value = newSession.id
    messagesMap.value.set(newSession.id, [])
  } catch (error) {
    console.error('创建会话失败:', error)
  }
}

/**
 * 删除会话
 * @param sessionId 会话ID
 */
const deleteSession = async (sessionId: string) => {
  try {
    await sessionApi.delete(sessionId)
    sessions.value = sessions.value.filter(s => s.id !== sessionId)
    messagesMap.value.delete(sessionId)
    // 如果删除的是当前会话，切换到第一个会话
    if (currentSessionId.value === sessionId) {
      currentSessionId.value = sessions.value[0]?.id || ''
    }
  } catch (error) {
    console.error('删除会话失败:', error)
  }
}

/**
 * 加载会话消息
 * @param sessionId 会话ID
 */
const loadSessionMessages = async (sessionId: string) => {
  try {
    const messages = await sessionApi.getMessages(sessionId)
    messagesMap.value.set(sessionId, messages)
  } catch (error) {
    console.error('加载消息失败:', error)
  }
}

// ==================== 消息发送方法 ====================

/**
 * 发送消息
 * @param content 消息内容
 */
const sendMessage = async (content: string) => {
  if (!content.trim()) return

  // 如果没有当前会话，先自动创建一个
  if (!currentSessionId.value) {
    await createSession()
    if (!currentSessionId.value) return // 创建失败
  }

  // 创建用户消息
  const userMessage: Message = {
    id: Date.now().toString(),
    role: 'user',
    content: content.trim(),
    timestamp: new Date().toISOString(),
  }

  // 添加用户消息到列表
  const messages = messagesMap.value.get(currentSessionId.value) || []
  messages.push(userMessage)
  messagesMap.value.set(currentSessionId.value, [...messages])

  // 设置加载状态
  isLoading.value = true

  try {
    // 创建AI消息占位符
    const aiMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      isStreaming: true,
    }
    messages.push(aiMessage)
    messagesMap.value.set(currentSessionId.value, [...messages])

    // 使用SSE流式接收响应
    await chatApi.sendMessageStream(
      currentSessionId.value,
      content.trim(),
      // 每次收到数据块时的回调
      (chunk: string) => {
        aiMessage.content += chunk
        messagesMap.value.set(currentSessionId.value, [...messages])
      },
      // 完成时的回调
      () => {
        aiMessage.isStreaming = false
        messagesMap.value.set(currentSessionId.value, [...messages])
        isLoading.value = false
      },
      // 错误时的回调
      (error: Error) => {
        console.error('流式响应错误:', error)
        aiMessage.content = '抱歉，发生错误，请重试。'
        aiMessage.isStreaming = false
        messagesMap.value.set(currentSessionId.value, [...messages])
        isLoading.value = false
      }
    )
  } catch (error) {
    console.error('发送消息失败:', error)
    isLoading.value = false
  }
}

// ==================== 技能执行方法 ====================

/**
 * 执行技能
 * @param skillId 技能ID
 */
const executeSkill = async (skillId: string) => {
  if (!currentSessionId.value) {
    warning('请先选择或创建一个会话')
    return
  }

  try {
    isLoading.value = true
    const result = await skillApi.execute(skillId, currentSessionId.value)

    // 将技能执行结果作为系统消息添加
    const systemMessage: Message = {
      id: Date.now().toString(),
      role: 'assistant',
      content: result.output || '技能执行完成',
      timestamp: new Date().toISOString(),
    }

    const messages = messagesMap.value.get(currentSessionId.value) || []
    messages.push(systemMessage)
    messagesMap.value.set(currentSessionId.value, [...messages])

    success('技能执行成功')
  } catch (err) {
    console.error('技能执行失败:', err)
    error('技能执行失败，请重试')
  } finally {
    isLoading.value = false
  }
}

/**
 * 刷新技能列表
 */
const refreshSkills = async () => {
  try {
    isLoading.value = true
    const skillsData = await skillApi.getAll()
    skills.value = skillsData
  } catch (error) {
    console.error('刷新技能列表失败:', error)
  } finally {
    isLoading.value = false
  }
}

// ==================== 初始化 ====================

/**
 * 组件挂载时加载数据
 */
onMounted(async () => {
  try {
    // 并行加载会话列表和技能列表
    const [sessionsData, skillsData] = await Promise.all([
      sessionApi.getAll(),
      skillApi.getAll(),
    ])

    sessions.value = sessionsData
    skills.value = skillsData

    // 如果有会话，选择第一个
    if (sessionsData.length > 0) {
      currentSessionId.value = sessionsData[0].id
      await loadSessionMessages(sessionsData[0].id)
    }
  } catch (error) {
    console.error('初始化失败:', error)
  }
})
</script>

<style scoped>
/* 应用容器 - 使用CSS Grid布局 */
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: var(--bg-primary);
  color: var(--text-primary);
}

/* 顶部导航栏 */
.app-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 24px;
  background-color: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
}

.app-brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.app-logo {
  color: var(--primary-color);
}

.app-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--primary-color);
}

.app-subtitle {
  font-size: 0.875rem;
  color: var(--text-secondary);
  flex: 1;
}

.app-actions {
  display: flex;
  gap: 8px;
}

.header-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: var(--radius-md);
  background-color: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
  cursor: pointer;
  transition: all 0.2s;
}

.header-btn:hover {
  background-color: var(--hover-bg);
  color: var(--primary-color);
  border-color: var(--primary-color);
}

/* 主内容区域 - 三栏布局 */
.app-main {
  display: grid;
  grid-template-columns: 280px 1fr 260px;
  flex: 1;
  overflow: hidden;
  transition: grid-template-columns 0.3s ease;
}

.app-main.sidebar-collapsed {
  grid-template-columns: 0 1fr 260px;
}

/* 左侧边栏 */
.sidebar {
  background-color: var(--bg-secondary);
  border-right: 1px solid var(--border-color);
  overflow-y: auto;
  transition: width 0.3s ease, opacity 0.3s ease;
}

.sidebar.collapsed {
  width: 0;
  opacity: 0;
  overflow: hidden;
}

/* 聊天区域 */
.chat-area {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* 技能面板 */
.skill-panel {
  background-color: var(--bg-secondary);
  border-left: 1px solid var(--border-color);
  overflow-y: auto;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .app-main {
    grid-template-columns: 240px 1fr;
  }

  .skill-panel {
    display: none;
  }

  .app-main.sidebar-collapsed {
    grid-template-columns: 0 1fr;
  }
}

@media (max-width: 768px) {
  .app-main {
    grid-template-columns: 1fr;
  }

  .sidebar {
    position: fixed;
    left: 0;
    top: 56px;
    bottom: 0;
    width: 280px;
    z-index: 100;
    transform: translateX(-100%);
    transition: transform 0.3s ease;
  }

  .sidebar.collapsed {
    transform: translateX(-100%);
    width: 280px;
    opacity: 1;
  }

  .sidebar:not(.collapsed) {
    transform: translateX(0);
  }
}
</style>
