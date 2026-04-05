<!--
  Towngas Manus Frontend - 聊天界面组件

  功能：
  1. 显示消息列表
  2. 消息输入框
  3. 发送按钮
  4. 流式消息支持
  5. 加载状态
  6. Markdown渲染

  已迁移到 Element Plus：
  - el-input type="textarea" 替代 textarea
  - el-icon + Loading 替代加载指示器
  - el-button type="primary" circle 替代发送按钮
-->

<template>
  <div class="chat-interface">
    <!-- 消息列表区域 -->
    <div class="message-list" ref="messageListRef">
      <!-- 空状态提示 -->
      <div v-if="messages.length === 0" class="empty-state">
        <div class="empty-icon">💬</div>
        <h3>开始对话</h3>
        <p>发送一条消息开始与AI助手交流</p>
      </div>

      <!-- 消息列表 -->
      <div
        v-for="message in messages"
        :key="message.id"
        :class="['message', `message-${message.role}`]"
      >
        <!-- 消息头像 -->
        <div class="message-avatar">
          <span v-if="message.role === 'user'">👤</span>
          <span v-else>🤖</span>
        </div>

        <!-- 消息内容 -->
        <div class="message-content">
          <!-- 工具调用列表 -->
          <div v-if="message.toolCalls && message.toolCalls.length > 0" class="tool-calls">
            <ToolCallIndicator
              v-for="toolCall in message.toolCalls"
              :key="toolCall.id"
              :id="toolCall.id"
              :name="toolCall.name"
              :status="toolCall.status"
              :input="toolCall.input"
              :output="toolCall.output"
              :error="toolCall.error"
              :duration="getToolCallDuration(toolCall)"
            />
          </div>

          <!-- 使用v-html渲染Markdown内容 -->
          <div
            class="message-text"
            ref="messageTextRef"
            v-html="renderMarkdown(message.content)"
          ></div>

          <!-- 流式加载指示器 -->
          <span v-if="message.isStreaming" class="streaming-indicator">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
          </span>

          <!-- 消息时间 -->
          <div v-if="!message.isStreaming" class="message-time">
            {{ formatTime(message.timestamp) }}
          </div>
        </div>
      </div>

      <!-- 加载状态指示器 -->
      <div v-if="isLoading && !hasStreamingMessage" class="loading-indicator">
        <el-icon class="is-loading" :size="20">
          <Loading />
        </el-icon>
        <span>AI正在思考...</span>
      </div>
    </div>

    <!-- 输入区域 -->
    <div class="input-area">
      <div class="input-container">
        <!-- Element Plus 多行文本输入框 -->
        <el-input
          v-model="inputText"
          type="textarea"
          class="message-input"
          :autosize="{ minRows: 1, maxRows: 6 }"
          placeholder="输入消息... (Shift+Enter换行，Enter发送)"
          :disabled="isLoading"
          @keydown="handleKeydown"
          ref="inputRef"
        />

        <!-- Element Plus 发送按钮 -->
        <el-button
          type="primary"
          circle
          class="send-button"
          :disabled="!canSend"
          @click="sendMessage"
          title="发送消息"
        >
          <el-icon v-if="!isLoading" :size="20">
            <Promotion />
          </el-icon>
          <el-icon v-else class="is-loading" :size="20">
            <Loading />
          </el-icon>
        </el-button>
      </div>

      <!-- 输入提示 -->
      <div class="input-hint">
        <span>按 Enter 发送，Shift+Enter 换行</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 聊天界面组件逻辑
 *
 * 使用Vue3 Composition API实现
 * 已迁移到 Element Plus 组件库
 */

import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import hljs from 'highlight.js'
// Element Plus 图标
import { Loading, Promotion } from '@element-plus/icons-vue'
import ToolCallIndicator from './ui/ToolCallIndicator.vue'
import type { Message, ToolCall } from '@/types'

// ==================== Props定义 ====================

interface Props {
  /** 消息列表 */
  messages: Message[]
  /** 是否正在加载 */
  isLoading: boolean
}

const props = defineProps<Props>()

// ==================== Emits定义 ====================

const emit = defineEmits<{
  /** 发送消息事件 */
  (e: 'send', content: string): void
}>()

// ==================== 状态 ====================

// 输入文本
const inputText = ref('')
// 消息列表DOM引用
const messageListRef = ref<HTMLElement | null>(null)
// 输入框DOM引用 - Element Plus InputInstance
const inputRef = ref<{ textarea: HTMLTextAreaElement } | null>(null)

// ==================== 计算属性 ====================

// 是否可以发送消息
const canSend = computed(() => {
  return inputText.value.trim().length > 0 && !props.isLoading
})

// 是否有正在流式传输的消息
const hasStreamingMessage = computed(() => {
  return props.messages.some((m) => m.isStreaming)
})

// ==================== Markdown渲染配置（模块级别只执行一次）====================

/**
 * 自定义代码高亮渲染器
 * 使用highlight.js进行代码高亮
 */
const createCodeRenderer = () => {
  return {
    // 代码块渲染方法
    code(code: string, language: string | undefined): string {
      // 如果指定了语言且该语言可用，使用指定语言高亮
      if (language && hljs.getLanguage(language)) {
        try {
          const highlighted = hljs.highlight(code, { language }).value
          return `<pre><code class="hljs language-${language}">${highlighted}</code></pre>`
        } catch {
          // 忽略错误，使用自动检测
        }
      }
      // 自动检测语言
      const highlighted = hljs.highlightAuto(code).value
      return `<pre><code class="hljs">${highlighted}</code></pre>`
    },
  }
}

// 配置marked选项（只执行一次）
const markedRenderer = new marked.Renderer()
Object.assign(markedRenderer, createCodeRenderer())
marked.use({
  renderer: markedRenderer,
  breaks: true,
  gfm: true,
})

/**
 * 渲染Markdown内容
 * 使用DOMPurify进行XSS过滤
 *
 * @param content 原始内容
 * @returns 安全的HTML内容
 */
const renderMarkdown = (content: string): string => {
  if (!content) return ''

  // 解析Markdown并清理HTML
  const rawHtml = marked.parse(content) as string
  return DOMPurify.sanitize(rawHtml, {
    ALLOWED_TAGS: [
      'p',
      'br',
      'strong',
      'em',
      'u',
      's',
      'code',
      'pre',
      'blockquote',
      'ul',
      'ol',
      'li',
      'a',
      'h1',
      'h2',
      'h3',
      'h4',
      'h5',
      'h6',
      'table',
      'thead',
      'tbody',
      'tr',
      'th',
      'td',
      'span',
      'div',
    ],
    ALLOWED_ATTR: ['href', 'title', 'class', 'id', 'target', 'rel'],
  })
}

// ==================== 方法 ====================

/**
 * 格式化时间戳
 *
 * @param timestamp ISO格式时间戳
 * @returns 格式化后的时间字符串
 */
const formatTime = (timestamp: string): string => {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
  })
}

/**
 * 处理键盘事件
 *
 * @param event 键盘事件
 */
const handleKeydown = (event: KeyboardEvent): void => {
  // Enter发送，Shift+Enter换行
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendMessage()
  }
}

/**
 * 发送消息
 */
const sendMessage = (): void => {
  const content = inputText.value.trim()
  if (content && !props.isLoading) {
    emit('send', content)
    inputText.value = ''
    // Element Plus textarea 会自动重置高度
  }
}

/**
 * 获取工具调用耗时
 */
const getToolCallDuration = (toolCall: ToolCall): number | undefined => {
  if (!toolCall.startTime || !toolCall.endTime) return undefined
  const start = new Date(toolCall.startTime).getTime()
  const end = new Date(toolCall.endTime).getTime()
  return end - start
}

// 复制状态
const copiedCode = ref<string | null>(null)

/**
 * 复制代码到剪贴板
 */
const copyCode = async (code: string, id: string): Promise<void> => {
  try {
    await navigator.clipboard.writeText(code)
    copiedCode.value = id
    setTimeout(() => {
      copiedCode.value = null
    }, 2000)
  } catch (err) {
    console.error('复制失败:', err)
  }
}

/**
 * 处理代码块，添加复制按钮
 */
const processCodeBlocks = (): void => {
  nextTick(() => {
    const codeBlocks = document.querySelectorAll('.message-text pre')
    codeBlocks.forEach((pre, index) => {
      // 检查是否已经处理过
      if (pre.querySelector('.code-copy-btn')) return

      const code = pre.querySelector('code')
      if (!code) return

      const codeText = code.textContent || ''
      const copyId = `code-${index}-${Date.now()}`

      // 创建复制按钮
      const copyBtn = document.createElement('button')
      copyBtn.className = 'code-copy-btn'
      copyBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      `
      copyBtn.setAttribute('data-code', codeText)
      copyBtn.setAttribute('data-id', copyId)
      copyBtn.onclick = async () => {
        const btnCode = copyBtn.getAttribute('data-code') || ''
        await copyCode(btnCode, copyId)
        copyBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        `
        setTimeout(() => {
          copyBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          `
        }, 2000)
      }

      // 设置 pre 为相对定位
      ;(pre as HTMLElement).style.position = 'relative'
      pre.appendChild(copyBtn)
    })
  })
}

/**
 * 滚动到底部
 */
const scrollToBottom = (): void => {
  nextTick(() => {
    if (messageListRef.value) {
      messageListRef.value.scrollTop = messageListRef.value.scrollHeight
    }
  })
}

// ==================== 监听器 ====================

// 监听消息列表变化，自动滚动到底部并处理代码块
watch(
  () => props.messages,
  () => {
    scrollToBottom()
    processCodeBlocks()
  },
  { deep: true }
)

// 监听加载状态变化
watch(
  () => props.isLoading,
  () => {
    scrollToBottom()
    if (!props.isLoading) {
      processCodeBlocks()
    }
  }
)

// 组件挂载后处理代码块
onMounted(() => {
  processCodeBlocks()
})
</script>

<style scoped>
/* 聊天界面容器 */
.chat-interface {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: var(--bg-primary);
}

/* 消息列表区域 */
.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  scroll-behavior: smooth;
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-secondary);
  text-align: center;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-state h3 {
  margin: 0 0 8px 0;
  color: var(--text-primary);
}

.empty-state p {
  margin: 0;
  font-size: 0.875rem;
}

/* 消息样式 */
.message {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 用户消息靠右 */
.message-user {
  flex-direction: row-reverse;
}

/* 消息头像 */
.message-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
  background-color: var(--bg-secondary);
}

/* 消息内容容器 */
.message-content {
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 16px;
  position: relative;
}

.message-user .message-content {
  background-color: var(--primary-color);
  color: white;
  border-bottom-right-radius: 4px;
}

.message-assistant .message-content {
  background-color: var(--bg-secondary);
  color: var(--text-primary);
  border-bottom-left-radius: 4px;
}

/* 消息文本 */
.message-text {
  line-height: 1.6;
  word-wrap: break-word;
}

/* 工具调用区域 */
.tool-calls {
  margin-bottom: 12px;
}

/* 代码块样式 */
.message-text :deep(pre) {
  background-color: var(--code-bg);
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 8px 0;
  position: relative;
}

.message-text :deep(code) {
  font-family: 'Fira Code', 'Monaco', monospace;
  font-size: 0.875rem;
}

/* 代码复制按钮 */
.message-text :deep(.code-copy-btn) {
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 4px 8px;
  background-color: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 4px;
  color: #f8f8f2;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s, background-color 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.message-text :deep(pre:hover .code-copy-btn) {
  opacity: 1;
}

.message-text :deep(.code-copy-btn:hover) {
  background-color: rgba(255, 255, 255, 0.2);
}

/* 内联代码 */
.message-text :deep(code:not(pre code)) {
  background-color: var(--code-inline-bg);
  padding: 2px 6px;
  border-radius: 4px;
}

/* 链接样式 */
.message-text :deep(a) {
  color: var(--link-color);
  text-decoration: none;
}

.message-text :deep(a:hover) {
  text-decoration: underline;
}

/* 流式加载指示器 */
.streaming-indicator {
  display: inline-flex;
  gap: 4px;
  margin-left: 8px;
}

.streaming-indicator .dot {
  width: 6px;
  height: 6px;
  background-color: currentColor;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
}

.streaming-indicator .dot:nth-child(1) {
  animation-delay: 0s;
}

.streaming-indicator .dot:nth-child(2) {
  animation-delay: 0.2s;
}

.streaming-indicator .dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes bounce {
  0%,
  80%,
  100% {
    transform: scale(0);
  }
  40% {
    transform: scale(1);
  }
}

/* 消息时间 */
.message-time {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-top: 4px;
  text-align: right;
}

/* 加载指示器 */
.loading-indicator {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  color: var(--text-secondary);
}

/* Element Plus 加载图标旋转动画 */
.loading-indicator .el-icon.is-loading {
  animation: rotating 2s linear infinite;
}

@keyframes rotating {
  to {
    transform: rotate(360deg);
  }
}

/* 输入区域 */
.input-area {
  padding: 16px 20px;
  background-color: var(--bg-secondary);
  border-top: 1px solid var(--border-color);
}

.input-container {
  display: flex;
  gap: 12px;
  align-items: flex-end;
}

/* 消息输入框 - Element Plus 覆盖 */
.message-input {
  flex: 1;
}

.message-input :deep(.el-textarea__inner) {
  padding: 12px 16px;
  border-radius: 12px;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-size: 1rem;
  line-height: 1.5;
  resize: none;
  box-shadow: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.message-input :deep(.el-textarea__inner:focus) {
  border-color: var(--primary-color);
  box-shadow: 0 0 0 3px var(--primary-color-alpha);
}

.message-input :deep(.el-textarea__inner:disabled) {
  background-color: var(--bg-disabled);
  cursor: not-allowed;
}

/* 发送按钮 - Element Plus 覆盖 */
.send-button {
  width: 48px;
  height: 48px;
  border-radius: 12px;
}

.send-button:hover:not(:disabled) {
  transform: scale(1.05);
}

.send-button:active:not(:disabled) {
  transform: scale(0.95);
}

/* 输入提示 */
.input-hint {
  margin-top: 8px;
  font-size: 0.75rem;
  color: var(--text-muted);
  text-align: right;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .message-content {
    max-width: 85%;
  }

  .input-area {
    padding: 12px;
  }
}
</style>
