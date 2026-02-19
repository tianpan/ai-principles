<!--
  GlobalDialog - 全局确认对话框容器

  配合 useConfirm composable 使用
-->

<template>
  <Teleport to="body">
    <Transition name="dialog">
      <div v-if="state.isOpen" class="dialog-overlay" @click.self="cancel">
        <div class="dialog-container">
          <div class="dialog-icon" :class="`dialog-icon-${state.variant}`">
            <AlertTriangle v-if="state.variant === 'danger'" :size="24" />
            <Info v-else-if="state.variant === 'info'" :size="24" />
            <HelpCircle v-else :size="24" />
          </div>

          <h3 class="dialog-title">{{ state.title }}</h3>

          <p v-if="state.message" class="dialog-message">{{ state.message }}</p>

          <div class="dialog-actions">
            <button class="dialog-btn dialog-btn-cancel" @click="cancel">
              {{ state.cancelText }}
            </button>
            <button
              :class="['dialog-btn', 'dialog-btn-confirm', `dialog-btn-${state.variant}`]"
              @click="confirm"
            >
              {{ state.confirmText }}
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'
import { useConfirm } from '@/composables/useConfirm'
import { AlertTriangle, Info, HelpCircle } from 'lucide-vue-next'

const { state, confirm, cancel } = useConfirm()

// 键盘事件处理
const handleKeydown = (event: KeyboardEvent) => {
  if (!state.value.isOpen) return

  if (event.key === 'Escape') {
    cancel()
  } else if (event.key === 'Enter') {
    confirm()
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
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

@keyframes dialogOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
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
</style>
