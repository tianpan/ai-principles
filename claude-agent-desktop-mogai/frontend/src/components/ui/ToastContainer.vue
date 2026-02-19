<!--
  ToastContainer - 全局 Toast 容器

  在 App.vue 中使用，管理所有 Toast 通知
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
import { useToast } from '@/composables/useToast'
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-vue-next'

const { toasts, removeToast } = useToast()
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
