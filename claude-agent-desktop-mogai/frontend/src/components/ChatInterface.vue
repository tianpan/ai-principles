<!--
  Towngas Manus Frontend - 聊天界面组件

  功能：
  1. 显示消息列表
  2. 消息输入框
  3. 发送按钮
  4. 流式消息支持
  5. 加载状态
  6. Markdown渲染
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
          <!-- 使用v-html渲染Markdown内容 -->
          <div
            class="message-text"
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
        <div class="loading-spinner"></div>
        <span>AI正在思考...</span>
      </div>
    </div>

    <!-- 输入区域 -->
    <div class="input-area">
      <div class="input-container">
        <!-- 多行文本输入框 -->
        <textarea
          v-model="inputText"
          class="message-input"
          placeholder="输入消息... (Shift+Enter换行，Enter发送)"
          rows="1"
          :disabled="isLoading"
          @keydown="handleKeydown"
          @input="autoResize"
          ref="inputRef"
        ></textarea>

        <!-- 发送按钮 -->
        <button
          class="send-button"
          :disabled="!canSend"
          @click="sendMessage"
          title="发送消息"
        >
          <span v-if="!isLoading">📤</span>
          <span v-else class="sending-spinner">⏳</span>
        </button>
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
 */

import { ref, computed, watch, nextTick } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import hljs from 'highlight.js'
import type { Message } from '@/types'

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
// 输入框DOM引用
const inputRef = ref<HTMLTextAreaElement | null>(null)

// ==================== 计算属性 ====================

// 是否可以发送消息
const canSend = computed(() => {
  return inputText.value.trim().length > 0 && !props.isLoading
})

// 是否有正在流式传输的消息
const hasStreamingMessage = computed(() => {
  return props.messages.some((m) => m.isStreaming)
})

// ==================== Markdown渲染配置 ====================

/**
 * 自定义代码高亮渲染器
 * 使用highlight.js进行代码高亮
 */
const renderer = {
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

// 配置marked选项
marked.setOptions({
  breaks: true, // 支持换行
  gfm: true, // 支持GitHub风格Markdown
})

// 应用自定义渲染器用于代码高亮
const markedRenderer = new marked.Renderer()
Object.assign(markedRenderer, renderer)
marked.use({ renderer: markedRenderer })

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
 * 自动调整输入框高度
 */
const autoResize = (): void => {
  const textarea = inputRef.value
  if (textarea) {
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
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
    // 重置输入框高度
    if (inputRef.value) {
      inputRef.value.style.height = 'auto'
    }
  }
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

// 监听消息列表变化，自动滚动到底部
watch(
  () => props.messages,
  () => {
    scrollToBottom()
  },
  { deep: true }
)

// 监听加载状态变化
watch(
  () => props.isLoading,
  () => {
    scrollToBottom()
  }
)
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

/* 代码块样式 */
.message-text :deep(pre) {
  background-color: var(--code-bg);
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 8px 0;
}

.message-text :deep(code) {
  font-family: 'Fira Code', 'Monaco', monospace;
  font-size: 0.875rem;
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

.loading-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--border-color);
  border-top-color: var(--primary-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
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

/* 消息输入框 */
.message-input {
  flex: 1;
  padding: 12px 16px;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-size: 1rem;
  line-height: 1.5;
  resize: none;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
  max-height: 200px;
  overflow-y: auto;
}

.message-input:focus {
  border-color: var(--primary-color);
  box-shadow: 0 0 0 3px var(--primary-color-alpha);
}

.message-input:disabled {
  background-color: var(--bg-disabled);
  cursor: not-allowed;
}

/* 发送按钮 */
.send-button {
  width: 48px;
  height: 48px;
  border: none;
  border-radius: 12px;
  background-color: var(--primary-color);
  color: white;
  font-size: 20px;
  cursor: pointer;
  transition: background-color 0.2s, transform 0.1s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.send-button:hover:not(:disabled) {
  background-color: var(--primary-color-dark);
  transform: scale(1.05);
}

.send-button:active:not(:disabled) {
  transform: scale(0.95);
}

.send-button:disabled {
  background-color: var(--bg-disabled);
  cursor: not-allowed;
}

.sending-spinner {
  animation: spin 1s linear infinite;
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
