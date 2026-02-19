/**
 * ConfirmDialog Composable
 *
 * 提供全局确认对话框功能
 */

import { ref } from 'vue'

export interface ConfirmOptions {
  title?: string
  message?: string
  confirmText?: string
  cancelText?: string
  variant?: 'default' | 'danger' | 'info'
}

interface ConfirmState {
  isOpen: boolean
  title: string
  message: string
  confirmText: string
  cancelText: string
  variant: 'default' | 'danger' | 'info'
}

// 全局状态
const state = ref<ConfirmState>({
  isOpen: false,
  title: '确认操作',
  message: '',
  confirmText: '确认',
  cancelText: '取消',
  variant: 'default',
})

let resolvePromise: ((value: boolean) => void) | null = null

/**
 * 显示确认对话框
 */
const show = (options: ConfirmOptions = {}): Promise<boolean> => {
  state.value = {
    isOpen: true,
    title: options.title || '确认操作',
    message: options.message || '',
    confirmText: options.confirmText || '确认',
    cancelText: options.cancelText || '取消',
    variant: options.variant || 'default',
  }

  return new Promise((resolve) => {
    resolvePromise = resolve
  })
}

/**
 * 确认
 */
const confirm = () => {
  state.value.isOpen = false
  if (resolvePromise) {
    resolvePromise(true)
    resolvePromise = null
  }
}

/**
 * 取消
 */
const cancel = () => {
  state.value.isOpen = false
  if (resolvePromise) {
    resolvePromise(false)
    resolvePromise = null
  }
}

/**
 * ConfirmDialog Composable
 */
export function useConfirm() {
  return {
    state,
    show,
    confirm,
    cancel,
  }
}

// 导出全局实例方法
export const confirmDialog = {
  show,
  danger: (message: string, title = '危险操作'): Promise<boolean> => {
    return show({ title, message, variant: 'danger', confirmText: '确认删除' })
  },
  info: (message: string, title = '提示'): Promise<boolean> => {
    return show({ title, message, variant: 'info' })
  },
}
