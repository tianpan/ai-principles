/**
 * Towngas Manus Frontend - TypeScript 类型定义
 *
 * 这个文件定义了整个应用中使用的所有TypeScript接口和类型
 */

// ==================== 消息相关类型 ====================

/**
 * 消息角色枚举
 */
export type MessageRole = 'user' | 'assistant' | 'system'

/**
 * 消息接口
 */
export interface Message {
  /** 消息唯一ID */
  id: string
  /** 消息角色 */
  role: MessageRole
  /** 消息内容 */
  content: string
  /** 消息时间戳 */
  timestamp: string
  /** 是否正在流式传输 */
  isStreaming?: boolean
  /** 附加元数据 */
  metadata?: MessageMetadata
}

/**
 * 消息元数据
 */
export interface MessageMetadata {
  /** 使用的模型 */
  model?: string
  /** Token消耗 */
  tokens?: number
  /** 响应时间（毫秒） */
  responseTime?: number
  /** 技能ID（如果是技能执行结果） */
  skillId?: string
}

// ==================== 会话相关类型 ====================

/**
 * 会话接口
 */
export interface Session {
  /** 会话唯一ID */
  id: string
  /** 会话标题 */
  title: string
  /** 创建时间 */
  createdAt: string
  /** 最后更新时间 */
  updatedAt: string
  /** 消息数量 */
  messageCount?: number
  /** 会话元数据 */
  metadata?: SessionMetadata
}

/**
 * 会话元数据
 */
export interface SessionMetadata {
  /** 使用的模型 */
  model?: string
  /** 总Token消耗 */
  totalTokens?: number
  /** 标签 */
  tags?: string[]
}

// ==================== 技能相关类型 ====================

/**
 * 技能状态枚举
 */
export type SkillStatus = 'available' | 'running' | 'error' | 'disabled'

/**
 * 技能接口
 */
export interface Skill {
  /** 技能唯一ID */
  id: string
  /** 技能名称 */
  name: string
  /** 技能描述 */
  description: string
  /** 技能图标 */
  icon?: string
  /** 技能状态 */
  status: SkillStatus
  /** 技能分类 */
  category?: string
  /** 技能参数定义 */
  parameters?: SkillParameter[]
}

/**
 * 技能参数定义
 */
export interface SkillParameter {
  /** 参数名 */
  name: string
  /** 参数类型 */
  type: 'string' | 'number' | 'boolean' | 'object' | 'array'
  /** 是否必填 */
  required: boolean
  /** 参数描述 */
  description?: string
  /** 默认值 */
  defaultValue?: unknown
}

/**
 * 技能执行结果
 */
export interface SkillResult {
  /** 是否成功 */
  success: boolean
  /** 输出内容 */
  output: string
  /** 错误信息 */
  error?: string
  /** 执行时间（毫秒） */
  executionTime?: number
}

// ==================== API响应类型 ====================

/**
 * 通用API响应包装
 */
export interface ApiResponse<T> {
  /** 是否成功 */
  success: boolean
  /** 响应数据 */
  data?: T
  /** 错误信息 */
  error?: string
  /** 错误代码 */
  errorCode?: string
}

/**
 * 分页参数
 */
export interface PaginationParams {
  /** 页码（从1开始） */
  page: number
  /** 每页数量 */
  limit: number
}

/**
 * 分页响应
 */
export interface PaginatedResponse<T> {
  /** 数据列表 */
  items: T[]
  /** 总数 */
  total: number
  /** 当前页码 */
  page: number
  /** 每页数量 */
  limit: number
  /** 总页数 */
  totalPages: number
}

// ==================== 流式响应类型 ====================

/**
 * SSE事件类型
 */
export type SSEEventType = 'message' | 'error' | 'done'

/**
 * SSE消息事件
 */
export interface SSEMessageEvent {
  /** 事件类型 */
  type: SSEEventType
  /** 数据内容 */
  data: string
  /** 事件ID */
  id?: string
}

// ==================== 错误类型 ====================

/**
 * API错误类
 */
export class ApiError extends Error {
  /** HTTP状态码 */
  statusCode: number
  /** 错误代码 */
  errorCode?: string

  constructor(message: string, statusCode: number, errorCode?: string) {
    super(message)
    this.name = 'ApiError'
    this.statusCode = statusCode
    this.errorCode = errorCode
  }
}
