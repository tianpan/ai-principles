<!--
  ConfirmDialog 确认对话框组件

  @deprecated 此组件已废弃，请使用 Element Plus 的 ElMessageBox
  迁移指南：
  - 使用 useConfirm() composable，它现在内部使用 ElMessageBox
  - API 保持向后兼容：show(), danger(), info()

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
          <div class="dialog-icon" :class="`dialog-icon-${currentVariant}`">
            <AlertTriangle v-if="currentVariant === 'danger'" :size="24" />
            <Info v-else-if="currentVariant === 'info'" :size="24" />
            <HelpCircle v-else :size="24" />
          </div>

          <h3 class="dialog-title">{{ currentTitle }}</h3>

          <p v-if="currentMessage" class="dialog-message">{{ currentMessage }}</p>

          <div class="dialog-actions">
            <button class="dialog-btn dialog-btn-cancel" @click="handleCancel">
              {{ currentCancelText }}
            </button>
            <button
              :class="['dialog-btn', 'dialog-btn-confirm', `dialog-btn-${currentVariant}`]"
              @click="handleConfirm"
            >
              {{ currentConfirmText }}
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

export interface ConfirmDialogOptions {
  title?: string
  message?: string
  confirmText?: string
  cancelText?: string
  variant?: 'default' | 'danger' | 'info'
}

// 内部状态（不直接修改 props）
const isOpen = ref(false)
const currentTitle = ref('确认操作')
const currentMessage = ref('')
const currentConfirmText = ref('确认')
const currentCancelText = ref('取消')
const currentVariant = ref<'default' | 'danger' | 'info'>('default')

let resolvePromise: ((value: boolean) => void) | null = null

/**
 * 显示对话框
 */
const show = (options?: ConfirmDialogOptions): Promise<boolean> => {
  // 更新内部状态（而不是修改 props）
  if (options) {
    currentTitle.value = options.title || '确认操作'
    currentMessage.value = options.message || ''
    currentConfirmText.value = options.confirmText || '确认'
    currentCancelText.value = options.cancelText || '取消'
    currentVariant.value = options.variant || 'default'
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
