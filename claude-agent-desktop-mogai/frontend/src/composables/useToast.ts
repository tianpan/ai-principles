/**
 * Toast Composable
 *
 * 使用 Element Plus ElMessage 实现的 Toast 通知功能
 * 保持向后兼容的 API
 */

import { ElMessage, type MessageParams } from 'element-plus'

export interface ToastOptions {
  message: string
  type?: 'success' | 'error' | 'warning' | 'info'
  duration?: number
}

interface ToastState {
  success: (message: string, duration?: number) => void
  error: (message: string, duration?: number) => void
  warning: (message: string, duration?: number) => void
  info: (message: string, duration?: number) => void
}

/**
 * 显示 Toast 消息
 */
const showToast = (options: ToastOptions): void => {
  const { message, type = 'info', duration = 4000 } = options

  const params: MessageParams = {
    message,
    duration,
    grouping: true,
    offset: 20,
  }

  switch (type) {
    case 'success':
      ElMessage.success(params)
      break
    case 'error':
      ElMessage.error(params)
      break
    case 'warning':
      ElMessage.warning(params)
      break
    case 'info':
    default:
      ElMessage.info(params)
      break
  }
}

/**
 * Toast Composable
 *
 * 使用 Element Plus ElMessage 实现
 */
export function useToast(): ToastState {
  return {
    success: (message: string, duration?: number) => showToast({ message, type: 'success', duration }),
    error: (message: string, duration?: number) => showToast({ message, type: 'error', duration }),
    warning: (message: string, duration?: number) => showToast({ message, type: 'warning', duration }),
    info: (message: string, duration?: number) => showToast({ message, type: 'info', duration }),
  }
}

// 导出全局实例方法（供组件外部使用）
// 保持与原有 API 的向后兼容
export const toast = {
  success: (message: string, duration?: number) => showToast({ message, type: 'success', duration }),
  error: (message: string, duration?: number) => showToast({ message, type: 'error', duration }),
  warning: (message: string, duration?: number) => showToast({ message, type: 'warning', duration }),
  info: (message: string, duration?: number) => showToast({ message, type: 'info', duration }),
}
