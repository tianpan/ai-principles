<!--
  Toast 通知组件

  符合 Apple HIG 设计规范：
  - 简洁、清晰的消息
  - 优雅的动画过渡
  - 自动消失机制
  - 支持多种类型（success, error, warning, info）
-->

<template>
  <Teleport to="body">
    <div class="toast-container">
      <TransitionGroup name="toast">
        <div
          v-for="toast in toasts"
          :key="toast.id"
          :class="['toast', `toast-${toast.type}`]"
          @click="removeToast(toast.id)"
        >
          <div class="toast-icon">
            <CheckCircle v-if="toast.type === 'success'" :size="18" />
            <XCircle v-else-if="toast.type === 'error'" :size="18" />
            <AlertTriangle v-else-if="toast.type === 'warning'" :size="18" />
            <Info v-else :size="18" />
          </div>
          <div class="toast-content">
            <span class="toast-message">{{ toast.message }}</span>
          </div>
          <button class="toast-close" @click.stop="removeToast(toast.id)">
            <X :size="14" />
          </button>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-vue-next'

export interface ToastItem {
  id: number
  message: string
  type: 'success' | 'error' | 'warning' | 'info'
  duration?: number
}

// Toast 列表
const toasts = ref<ToastItem[]>([])
let toastId = 0

/**
 * 添加 Toast 通知
 */
const addToast = (message: string, type: ToastItem['type'] = 'info', duration = 4000) => {
  const id = ++toastId
  const toast: ToastItem = { id, message, type, duration }
  toasts.value.push(toast)

  // 自动移除
  if (duration > 0) {
    setTimeout(() => {
      removeToast(id)
    }, duration)
  }

  return id
}

/**
 * 移除 Toast
 */
const removeToast = (id: number) => {
  const index = toasts.value.findIndex(t => t.id === id)
  if (index > -1) {
    toasts.value.splice(index, 1)
  }
}

// 便捷方法
const success = (message: string, duration?: number) => addToast(message, 'success', duration)
const error = (message: string, duration?: number) => addToast(message, 'error', duration)
const warning = (message: string, duration?: number) => addToast(message, 'warning', duration)
const info = (message: string, duration?: number) => addToast(message, 'info', duration)

// 暴露给 composable 使用
defineExpose({
  addToast,
  removeToast,
  success,
  error,
  warning,
  info
})

// 清理
onUnmounted(() => {
  toasts.value = []
})
</script>

<style scoped>
.toast-container {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
}

.toast {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background-color: var(--bg-primary);
  border-radius: var(--radius-md);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  border-left: 3px solid var(--primary-color);
  min-width: 280px;
  max-width: 400px;
  pointer-events: auto;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.toast:hover {
  transform: translateX(-4px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
}

.toast-success { border-left-color: var(--success-color); }
.toast-error { border-left-color: var(--danger-color); }
.toast-warning { border-left-color: var(--warning-color); }
.toast-info { border-left-color: var(--primary-color); }

.toast-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.toast-success .toast-icon { color: var(--success-color); }
.toast-error .toast-icon { color: var(--danger-color); }
.toast-warning .toast-icon { color: var(--warning-color); }
.toast-info .toast-icon { color: var(--primary-color); }

.toast-content {
  flex: 1;
  min-width: 0;
}

.toast-message {
  font-size: 0.875rem;
  color: var(--text-primary);
  line-height: 1.4;
}

.toast-close {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  transition: background-color 0.2s, color 0.2s;
  flex-shrink: 0;
}

.toast-close:hover {
  background-color: var(--hover-bg);
  color: var(--text-primary);
}

/* 动画 */
.toast-enter-active {
  animation: toastIn 0.3s ease;
}

.toast-leave-active {
  animation: toastOut 0.2s ease;
}

@keyframes toastIn {
  from {
    opacity: 0;
    transform: translateX(100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes toastOut {
  from {
    opacity: 1;
    transform: translateX(0);
  }
  to {
    opacity: 0;
    transform: translateX(100%);
  }
}
</style>
