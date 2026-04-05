/**
 * ConfirmDialog Composable
 *
 * 使用 Element Plus ElMessageBox 实现的确认对话框功能
 * 保持向后兼容的 API
 */

import { ElMessageBox } from 'element-plus'

export interface ConfirmOptions {
  title?: string
  message?: string
  confirmText?: string
  cancelText?: string
  variant?: 'default' | 'danger' | 'info'
}

interface ConfirmState {
  show: (options?: ConfirmOptions) => Promise<boolean>
  danger: (message: string, title?: string) => Promise<boolean>
  info: (message: string, title?: string) => Promise<boolean>
}

/**
 * 显示确认对话框
 */
const show = async (options: ConfirmOptions = {}): Promise<boolean> => {
  const {
    title = '确认操作',
    message = '',
    confirmText = '确认',
    cancelText = '取消',
    variant = 'default',
  } = options

  const confirmButtonText = confirmText
  const cancelButtonText = cancelText
  const type: 'warning' | 'info' = variant === 'danger' ? 'warning' : 'info'

  try {
    await ElMessageBox.confirm(message, title, {
      confirmButtonText,
      cancelButtonText,
      type,
      closeOnClickModal: false,
      closeOnPressEscape: true,
    })
    return true
  } catch {
    return false
  }
}

/**
 * ConfirmDialog Composable
 *
 * 使用 Element Plus ElMessageBox 实现
 */
export function useConfirm(): ConfirmState {
  return {
    show,
    danger: (message: string, title = '危险操作'): Promise<boolean> => {
      return show({ title, message, variant: 'danger', confirmText: '确认删除' })
    },
    info: (message: string, title = '提示'): Promise<boolean> => {
      return show({ title, message, variant: 'info' })
    },
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
