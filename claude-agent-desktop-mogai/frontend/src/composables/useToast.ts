/**
 * Toast Composable
 *
 * 提供全局 Toast 通知功能
 */

import { ref, type Ref } from 'vue'

export interface ToastOptions {
  message: string
  type?: 'success' | 'error' | 'warning' | 'info'
  duration?: number
}

interface ToastState {
  toasts: Ref<Array<{
    id: number
    message: string
    type: 'success' | 'error' | 'warning' | 'info'
    duration: number
  }>>
  addToast: (options: ToastOptions) => number
  removeToast: (id: number) => void
  success: (message: string, duration?: number) => number
  error: (message: string, duration?: number) => number
  warning: (message: string, duration?: number) => number
  info: (message: string, duration?: number) => number
}

// 全局状态
const toasts = ref<Array<{
  id: number
  message: string
  type: 'success' | 'error' | 'warning' | 'info'
  duration: number
}>>([])

let toastId = 0

/**
 * 添加 Toast
 */
const addToast = (options: ToastOptions): number => {
  const id = ++toastId
  const { message, type = 'info', duration = 4000 } = options

  toasts.value.push({ id, message, type, duration })

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
const removeToast = (id: number): void => {
  const index = toasts.value.findIndex(t => t.id === id)
  if (index > -1) {
    toasts.value.splice(index, 1)
  }
}

/**
 * Toast Composable
 */
export function useToast(): ToastState {
  return {
    toasts,
    addToast,
    removeToast,
    success: (message: string, duration?: number) => addToast({ message, type: 'success', duration }),
    error: (message: string, duration?: number) => addToast({ message, type: 'error', duration }),
    warning: (message: string, duration?: number) => addToast({ message, type: 'warning', duration }),
    info: (message: string, duration?: number) => addToast({ message, type: 'info', duration }),
  }
}

// 导出全局实例方法（供组件外部使用）
export const toast = {
  success: (message: string, duration?: number) => addToast({ message, type: 'success', duration }),
  error: (message: string, duration?: number) => addToast({ message, type: 'error', duration }),
  warning: (message: string, duration?: number) => addToast({ message, type: 'warning', duration }),
  info: (message: string, duration?: number) => addToast({ message, type: 'info', duration }),
}
