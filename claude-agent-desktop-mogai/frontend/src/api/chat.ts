/**
 * Towngas Manus Frontend - API调用模块
 *
 * 这个模块封装了所有与后端API的交互，包括：
 * 1. 普通HTTP请求（使用Axios）
 * 2. SSE流式响应处理
 * 3. 错误处理
 */

import axios, { type AxiosInstance, type AxiosError } from 'axios'
import type { Session, Message, Skill, SkillResult, MessageRole, SkillStatus } from '@/types'
import { ApiError } from '@/types'

// ==================== 后端响应类型定义 ====================

/** 分页响应格式 */
interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  has_more: boolean
}

/** 后端会话格式 */
interface BackendSession {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  metadata: Record<string, unknown> | null
}

/** 后端会话详情格式 */
interface BackendSessionDetail extends BackendSession {
  messages: Array<{
    role: string
    content: string
    timestamp: string
    metadata: Record<string, unknown> | null
  }>
}

/** 后端聊天响应格式 */
interface BackendChatResponse {
  session_id: string
  message: {
    role: string
    content: string
    timestamp: string
  }
  finish_reason: string
}

/** 后端技能信息格式 */
interface BackendSkillInfo {
  name: string
  description: string
  category: string
  parameters: {
    type: string
    properties?: Record<string, {
      type?: string
      description?: string
      default?: unknown
    }>
    required?: string[]
  }
  metadata: Record<string, unknown> | null
}

/** 后端技能执行响应格式 */
interface BackendSkillExecuteResponse {
  skill_name: string
  success: boolean
  result: unknown
  error: string | null
}

// ==================== Axios实例配置 ====================

const httpClient: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 60000, // 60秒超时
  headers: {
    'Content-Type': 'application/json',
  },
})

httpClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    console.log('[API] 请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

httpClient.interceptors.response.use(
  (response) => {
    console.log('[API] 响应:', response.config.url, response.status)
    return response
  },
  (error: AxiosError) => {
    console.error('[API] 错误:', error.config?.url, error.message)
    const message = (error.response?.data as { detail?: string })?.detail || error.message || '请求失败'
    const statusCode = error.response?.status || 500
    return Promise.reject(new ApiError(message, statusCode))
  }
)

// ==================== SSE流式响应处理 ====================

function createSSEConnection(
  url: string,
  body: object,
  onMessage: (chunk: string) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): () => void {
  const abortController = new AbortController()

  console.log('[SSE] 连接:', url, body)

  fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(body),
    signal: abortController.signal,
  })
    .then(async (response) => {
      console.log('[SSE] 响应状态:', response.status)

      if (!response.ok) {
        // 尝试解析错误信息
        try {
          const errorData = await response.json()
          throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
        } catch {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('无法获取响应流')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()

        if (done) {
          if (buffer.trim()) {
            processSSELine(buffer, onMessage, onComplete, onError)
          }
          console.log('[SSE] 流结束')
          onComplete()
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          processSSELine(line, onMessage, onComplete, onError)
        }
      }
    })
    .catch((error) => {
      console.error('[SSE] 错误:', error)
      if (error.name !== 'AbortError') {
        onError(error)
      }
    })

  return () => {
    console.log('[SSE] 中止连接')
    abortController.abort()
  }
}

function processSSELine(
  line: string,
  onMessage: (chunk: string) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): void {
  // 跳过空行和注释
  if (!line || line.startsWith(':')) {
    return
  }

  // 处理 data: 前缀的数据行
  if (line.startsWith('data:')) {
    const data = line.slice(5).trim()

    // 检查是否是结束标记
    if (data === '[DONE]') {
      onComplete()
      return
    }

    try {
      const parsed = JSON.parse(data)
      console.log('[SSE] 事件:', parsed.type, parsed)

      // 根据后端返回的事件类型处理
      switch (parsed.type) {
        case 'text':
          if (parsed.content) {
            onMessage(parsed.content)
          }
          break

        case 'session_start':
          console.log('[SSE] 会话开始:', parsed.session_id)
          break

        case 'tool_use':
          console.log('[SSE] 工具调用:', parsed.tool_name)
          break

        case 'tool_result':
          console.log('[SSE] 工具结果:', parsed.tool_name)
          break

        case 'done':
          console.log('[SSE] 完成')
          onComplete()
          break

        case 'error':
          console.error('[SSE] 服务器错误:', parsed.error)
          onError(new Error(parsed.error || '服务器返回错误'))
          break

        default:
          // 兼容其他格式
          if (parsed.content) {
            onMessage(parsed.content)
          } else if (typeof parsed === 'string') {
            onMessage(parsed)
          }
      }
    } catch (e) {
      // 如果不是 JSON，直接作为文本处理
      console.log('[SSE] 文本数据:', data)
      onMessage(data)
    }
  }
}

// ==================== 工具函数：转换后端格式到前端格式 ====================

function convertSession(backend: BackendSession): Session {
  return {
    id: backend.id,
    title: backend.title,
    createdAt: backend.created_at,
    updatedAt: backend.updated_at,
    messageCount: backend.message_count,
    metadata: backend.metadata || undefined
  }
}

function convertMessages(
  sessionId: string,
  messages: BackendSessionDetail['messages']
): Message[] {
  return messages.map((msg, index) => ({
    id: `${sessionId}-${index}`,
    role: msg.role as MessageRole,
    content: msg.content,
    timestamp: msg.timestamp,
    metadata: msg.metadata || undefined
  }))
}

function convertSkill(backend: BackendSkillInfo): Skill {
  const properties = backend.parameters?.properties || {}
  const required = backend.parameters?.required || []

  return {
    id: backend.name,
    name: backend.name,
    description: backend.description,
    status: 'available' as SkillStatus,
    category: backend.category,
    parameters: Object.entries(properties).map(([name, prop]) => ({
      name,
      type: (prop?.type || 'string') as 'string' | 'number' | 'boolean' | 'object' | 'array',
      required: required.includes(name),
      description: prop?.description,
      defaultValue: prop?.default
    }))
  }
}

// ==================== 聊天API ====================

export const chatApi = {
  sendMessageStream(
    sessionId: string,
    content: string,
    onChunk: (chunk: string) => void,
    onComplete: () => void,
    onError: (error: Error) => void
  ): () => void {
    return createSSEConnection(
      '/api/chat/stream',
      { session_id: sessionId || null, message: content },
      onChunk,
      onComplete,
      onError
    )
  },

  async sendMessage(sessionId: string, content: string): Promise<string> {
    const response = await httpClient.post<BackendChatResponse>(
      '/api/chat',
      { session_id: sessionId || null, message: content }
    )
    return response.data.message?.content || ''
  },
}

// ==================== 会话API ====================

export const sessionApi = {
  async getAll(): Promise<Session[]> {
    const response = await httpClient.get<PaginatedResponse<BackendSession>>('/api/sessions')
    return (response.data.items || []).map(convertSession)
  },

  async getById(sessionId: string): Promise<Session> {
    const response = await httpClient.get<BackendSessionDetail>(
      `/api/sessions/${sessionId}`
    )
    if (!response.data || !response.data.id) {
      throw new ApiError('会话不存在', 404)
    }
    return convertSession(response.data)
  },

  async create(title?: string): Promise<Session> {
    const response = await httpClient.post<BackendSession>('/api/sessions', {
      title: title || '新会话',
    })
    if (!response.data || !response.data.id) {
      throw new ApiError('创建会话失败', 500)
    }
    return convertSession(response.data)
  },

  async update(sessionId: string, data: Partial<Session>): Promise<Session> {
    const response = await httpClient.put<BackendSession>(
      `/api/sessions/${sessionId}`,
      {
        title: data.title,
        metadata: data.metadata
      }
    )
    if (!response.data || !response.data.id) {
      throw new ApiError('更新会话失败', 500)
    }
    return convertSession(response.data)
  },

  async delete(sessionId: string): Promise<void> {
    await httpClient.delete(`/api/sessions/${sessionId}`)
  },

  async getMessages(sessionId: string): Promise<Message[]> {
    const response = await httpClient.get<BackendSessionDetail>(
      `/api/sessions/${sessionId}`
    )
    return convertMessages(sessionId, response.data.messages || [])
  },
}

// ==================== 技能API ====================

export const skillApi = {
  async getAll(): Promise<Skill[]> {
    const response = await httpClient.get<BackendSkillInfo[]>('/api/skills')
    return (response.data || []).map(convertSkill)
  },

  async getById(skillId: string): Promise<Skill> {
    const response = await httpClient.get<BackendSkillInfo>(
      `/api/skills/${skillId}`
    )
    if (!response.data || !response.data.name) {
      throw new ApiError('技能不存在', 404)
    }
    return convertSkill(response.data)
  },

  async execute(
    skillId: string,
    _sessionId: string,
    parameters?: Record<string, unknown>
  ): Promise<SkillResult> {
    const response = await httpClient.post<BackendSkillExecuteResponse>(
      `/api/skills/${skillId}/execute`,
      { arguments: parameters || {} }
    )
    return {
      success: response.data.success,
      output: response.data.result ? String(response.data.result) : '',
      error: response.data.error || undefined
    }
  },
}

export { httpClient }
