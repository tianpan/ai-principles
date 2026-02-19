<!--
  ConfirmDialog 确认对话框组件

  符合 Apple HIG 设计规范：
  - 清晰的标题和描述
  - 明确的操作按钮
  - 优雅的动画过渡
  - 键盘支持（Enter 确认，Escape 取消）
-->

<template>
  <Teleport to="body">
    <Transition name="dialog">
      <div v-if="isOpen" class="dialog-overlay" @click.self="handleCancel">
        <div class="dialog-container">
          <div class="dialog-icon" :class="`dialog-icon-${variant}`">
            <AlertTriangle v-if="variant === 'danger'" :size="24" />
            <Info v-else-if="variant === 'info'" :size="24" />
            <HelpCircle v-else :size="24" />
          </div>

          <h3 class="dialog-title">{{ title }}</h3>

          <p v-if="message" class="dialog-message">{{ message }}</p>

          <div class="dialog-actions">
            <button class="dialog-btn dialog-btn-cancel" @click="handleCancel">
              {{ cancelText }}
            </button>
            <button
              :class="['dialog-btn', 'dialog-btn-confirm', `dialog-btn-${variant}`]"
              @click="handleConfirm"
            >
              {{ confirmText }}
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { AlertTriangle, Info, HelpCircle } from 'lucide-vue-next'

export interface ConfirmDialogProps {
  title?: string
  message?: string
  confirmText?: string
  cancelText?: string
  variant?: 'default' | 'danger' | 'info'
}

const props = withDefaults(defineProps<ConfirmDialogProps>(), {
  title: '确认操作',
  message: '',
  confirmText: '确认',
  cancelText: '取消',
  variant: 'default',
})

const isOpen = ref(false)
let resolvePromise: ((value: boolean) => void) | null = null

/**
 * 显示对话框
 */
const show = (options?: Partial<ConfirmDialogProps>): Promise<boolean> => {
  // 更新 props（如果提供了选项）
  if (options) {
    Object.assign(props, options)
  }

  isOpen.value = true

  return new Promise((resolve) => {
    resolvePromise = resolve
  })
}

/**
 * 隐藏对话框
 */
const hide = () => {
  isOpen.value = false
  resolvePromise = null
}

/**
 * 处理确认
 */
const handleConfirm = () => {
  if (resolvePromise) {
    resolvePromise(true)
  }
  hide()
}

/**
 * 处理取消
 */
const handleCancel = () => {
  if (resolvePromise) {
    resolvePromise(false)
  }
  hide()
}

/**
 * 键盘事件处理
 */
const handleKeydown = (event: KeyboardEvent) => {
  if (!isOpen.value) return

  if (event.key === 'Escape') {
    handleCancel()
  } else if (event.key === 'Enter') {
    handleConfirm()
  }
}

// 监听键盘事件
onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})

// 暴露方法
defineExpose({
  show,
  hide,
  confirm: handleConfirm,
  cancel: handleCancel,
})
</script>

<style scoped>
.dialog-overlay {
  position: fixed;
  inset: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10001;
  backdrop-filter: blur(4px);
}

.dialog-container {
  background-color: var(--bg-primary);
  border-radius: var(--radius-lg);
  padding: 24px;
  min-width: 320px;
  max-width: 420px;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);
  text-align: center;
}

.dialog-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  border-radius: 50%;
  margin: 0 auto 16px;
}

.dialog-icon-default {
  background-color: var(--primary-color-alpha);
  color: var(--primary-color);
}

.dialog-icon-danger {
  background-color: var(--danger-color-alpha);
  color: var(--danger-color);
}

.dialog-icon-info {
  background-color: var(--primary-color-alpha);
  color: var(--primary-color);
}

.dialog-title {
  margin: 0 0 8px;
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--text-primary);
}

.dialog-message {
  margin: 0 0 24px;
  font-size: 0.875rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

.dialog-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
}

.dialog-btn {
  flex: 1;
  padding: 10px 20px;
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.dialog-btn-cancel {
  background-color: var(--bg-secondary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
}

.dialog-btn-cancel:hover {
  background-color: var(--hover-bg);
}

.dialog-btn-confirm {
  color: white;
  border: none;
}

.dialog-btn-default {
  background-color: var(--primary-color);
}

.dialog-btn-default:hover {
  background-color: var(--primary-color-dark);
}

.dialog-btn-danger {
  background-color: var(--danger-color);
}

.dialog-btn-danger:hover {
  background-color: #dc2626;
}

.dialog-btn-info {
  background-color: var(--primary-color);
}

.dialog-btn-info:hover {
  background-color: var(--primary-color-dark);
}

/* 动画 */
.dialog-enter-active {
  animation: dialogIn 0.2s ease;
}

.dialog-leave-active {
  animation: dialogOut 0.15s ease;
}

@keyframes dialogIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.dialog-enter-active .dialog-container {
  animation: dialogSlideIn 0.2s ease;
}

.dialog-leave-active .dialog-container {
  animation: dialogSlideOut 0.15s ease;
}

@keyframes dialogSlideIn {
  from {
    opacity: 0;
    transform: scale(0.95) translateY(-10px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

@keyframes dialogSlideOut {
  from {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
  to {
    opacity: 0;
    transform: scale(0.95) translateY(-10px);
  }
}

.dialog-leave-active .dialog-container {
  animation: dialogSlideOut 0.15s ease;
}

@keyframes dialogOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
</style>
