/**
 * Keyboard Shortcuts Composable
 *
 * 提供全局键盘快捷键功能
 * 支持 Cmd (Mac) / Ctrl (Windows) 组合键
 */

import { onMounted, onUnmounted } from 'vue'

export interface KeyboardShortcut {
  key: string
  ctrl?: boolean
  meta?: boolean
  shift?: boolean
  alt?: boolean
  handler: (event: KeyboardEvent) => void
  description?: string
}

/**
 * 判断是否是 macOS
 */
const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0

/**
 * 检查快捷键是否匹配
 */
const matchesShortcut = (event: KeyboardEvent, shortcut: KeyboardShortcut): boolean => {
  const key = event.key.toLowerCase()
  const targetKey = shortcut.key.toLowerCase()

  // 检查按键
  if (key !== targetKey) return false

  // Mac 使用 metaKey (Cmd)，Windows 使用 ctrlKey
  const ctrlPressed = isMac ? event.metaKey : event.ctrlKey

  // 检查修饰键 - 默认需要 Ctrl/Cmd
  const needsCtrl = shortcut.ctrl || (shortcut.meta === undefined && !isMac)
  if (needsCtrl && !ctrlPressed) return false
  if (shortcut.shift && !event.shiftKey) return false
  if (shortcut.alt && !event.altKey) return false

  return true
}

/**
 * Keyboard Shortcuts Composable
 */
export function useKeyboard(shortcuts: KeyboardShortcut[]) {
  const handleKeydown = (event: KeyboardEvent) => {
    // 如果焦点在输入框中，忽略单字符快捷键
    const target = event.target as HTMLElement
    const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable

    for (const shortcut of shortcuts) {
      if (matchesShortcut(event, shortcut)) {
        // 如果是输入框，只响应带有 Ctrl/Cmd 的快捷键
        if (isInput && !shortcut.ctrl && !shortcut.meta) continue

        event.preventDefault()
        shortcut.handler(event)
        return
      }
    }
  }

  onMounted(() => {
    document.addEventListener('keydown', handleKeydown)
  })

  onUnmounted(() => {
    document.removeEventListener('keydown', handleKeydown)
  })
}

/**
 * 预定义的快捷键
 */
export const SHORTCUTS = {
  NEW_SESSION: { key: 'n', ctrl: true, description: '新建会话' },
  SEARCH: { key: 'k', ctrl: true, description: '搜索' },
  TOGGLE_SIDEBAR: { key: '\\', ctrl: true, description: '切换侧边栏' },
  HELP: { key: '/', ctrl: true, description: '帮助' },
  CLOSE: { key: 'Escape', description: '关闭/取消' },
} as const
