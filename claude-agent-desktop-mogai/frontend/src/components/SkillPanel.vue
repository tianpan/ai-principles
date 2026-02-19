<!--
  Towngas Manus Frontend - 技能面板组件

  功能：
  1. 显示技能列表
  2. 技能分类显示
  3. 技能执行按钮
  4. 技能状态显示
-->

<template>
  <div class="skill-panel">
    <!-- 头部 -->
    <div class="skill-header">
      <h2 class="skill-title">技能面板</h2>
      <button
        class="refresh-btn"
        @click="handleRefresh"
        title="刷新技能列表"
      >
        <RefreshCw :size="16" />
      </button>
    </div>

    <!-- 技能列表 -->
    <div class="skill-list">
      <!-- 空状态 -->
      <div v-if="skills.length === 0" class="empty-skills">
        <Zap class="empty-icon" :size="32" />
        <p>暂无可用技能</p>
      </div>

      <!-- 按分类分组显示技能 -->
      <div v-else>
        <div
          v-for="group in groupedSkills"
          :key="group.category"
          class="skill-group"
        >
          <!-- 分类标题 -->
          <div class="group-title">
            {{ group.category || '其他' }}
            <span class="group-count">({{ group.skills.length }})</span>
          </div>

          <!-- 分类下的技能 -->
          <div
            v-for="skill in group.skills"
            :key="skill.id"
            class="skill-item"
          >
            <!-- 技能图标和名称 -->
            <div class="skill-main">
              <span class="skill-icon">
                <component :is="getSkillIcon(skill.id)" :size="16" />
              </span>
              <div class="skill-info">
                <div class="skill-name">{{ skill.name }}</div>
                <div class="skill-desc">{{ skill.description }}</div>
              </div>
            </div>

            <!-- 状态标签 -->
            <div :class="['skill-status', `status-${skill.status}`]">
              {{ getStatusText(skill.status) }}
            </div>

            <!-- 执行按钮 -->
            <button
              class="execute-btn"
              :disabled="skill.status !== 'available'"
              @click="handleExecute(skill.id)"
              title="执行技能"
            >
              <Play :size="14" />
              <span>执行</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 技能面板组件逻辑
 */

import { computed } from 'vue'
import {
  RefreshCw,
  Zap,
  Play,
  Clock,
  Calendar,
  Database,
  FileText,
  Activity,
  type LucideIcon
} from 'lucide-vue-next'
import type { Skill, SkillStatus } from '@/types'

// ==================== Props定义 ====================

interface Props {
  /** 技能列表 */
  skills: Skill[]
}

const props = defineProps<Props>()

// ==================== Emits定义 ====================

const emit = defineEmits<{
  /** 执行技能事件 */
  (e: 'execute', skillId: string): void
  /** 刷新技能列表事件 */
  (e: 'refresh'): void
}>()

// ==================== 计算属性 ====================

// 按分类分组的技能
interface SkillGroup {
  category: string
  skills: Skill[]
}

const groupedSkills = computed((): SkillGroup[] => {
  const groups: Map<string, Skill[]> = new Map()

  // 遍历技能，按分类分组
  props.skills.forEach((skill) => {
    const category = skill.category || '其他'
    if (!groups.has(category)) {
      groups.set(category, [])
    }
    groups.get(category)!.push(skill)
  })

  // 转换为数组
  return Array.from(groups.entries()).map(([category, skills]) => ({
    category,
    skills,
  }))
})

// ==================== 方法 ====================

/**
 * 获取技能图标
 *
 * @param skillId 技能ID
 * @returns 图标组件
 */
const getSkillIcon = (skillId: string): LucideIcon => {
  const iconMap: Record<string, LucideIcon> = {
    'get_current_time': Clock,
    'query_station': Database,
    'generate_report': FileText,
    'check_device': Activity,
    'schedule_task': Calendar,
  }
  return iconMap[skillId] || Zap
}

/**
 * 获取状态文本
 *
 * @param status 技能状态
 * @returns 状态文本
 */
const getStatusText = (status: SkillStatus): string => {
  const statusMap: Record<SkillStatus, string> = {
    available: '可用',
    running: '运行中',
    error: '错误',
    disabled: '已禁用',
  }
  return statusMap[status] || status
}

/**
 * 处理技能执行
 *
 * @param skillId 技能ID
 */
const handleExecute = (skillId: string): void => {
  emit('execute', skillId)
}

/**
 * 处理刷新
 */
const handleRefresh = (): void => {
  emit('refresh')
}
</script>

<style scoped>
/* 技能面板容器 */
.skill-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
}

/* 头部 */
.skill-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--border-color);
}

.skill-title {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

/* 刷新按钮 */
.refresh-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 8px;
  background-color: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  transition: background-color 0.2s, transform 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.refresh-btn:hover {
  background-color: var(--hover-bg);
  color: var(--primary-color);
}

.refresh-btn:active {
  transform: rotate(180deg);
}

/* 技能列表 */
.skill-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

/* 空状态 */
.empty-skills {
  padding: 40px 16px;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.875rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.empty-icon {
  opacity: 0.5;
}

/* 技能分组 */
.skill-group {
  margin-bottom: 16px;
}

/* 分组标题 */
.group-title {
  padding: 8px 12px;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.group-count {
  font-weight: 400;
  margin-left: 4px;
}

/* 技能项 */
.skill-item {
  display: flex;
  flex-direction: column;
  padding: 12px;
  border-radius: 8px;
  background-color: var(--bg-primary);
  margin-bottom: 8px;
  transition: box-shadow 0.2s;
}

.skill-item:hover {
  box-shadow: 0 2px 8px var(--shadow-color);
}

/* 技能主体 */
.skill-main {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
}

/* 技能图标 */
.skill-icon {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  background-color: var(--primary-color-alpha);
  color: var(--primary-color);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

/* 技能信息 */
.skill-info {
  flex: 1;
  min-width: 0;
}

.skill-name {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.skill-desc {
  font-size: 0.75rem;
  color: var(--text-secondary);
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* 状态标签 */
.skill-status {
  font-size: 0.625rem;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 500;
  margin-bottom: 8px;
  width: fit-content;
}

.status-available {
  background-color: var(--success-color-alpha);
  color: var(--success-color);
}

.status-running {
  background-color: var(--warning-color-alpha);
  color: var(--warning-color);
}

.status-error {
  background-color: var(--danger-color-alpha);
  color: var(--danger-color);
}

.status-disabled {
  background-color: var(--bg-secondary);
  color: var(--text-muted);
}

/* 执行按钮 */
.execute-btn {
  width: 100%;
  padding: 8px 12px;
  border: none;
  border-radius: 6px;
  background-color: var(--primary-color);
  color: white;
  cursor: pointer;
  transition: background-color 0.2s, transform 0.1s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-size: 0.875rem;
  font-weight: 500;
}

.execute-btn:hover:not(:disabled) {
  background-color: var(--primary-color-dark);
  transform: scale(1.02);
}

.execute-btn:active:not(:disabled) {
  transform: scale(0.98);
}

.execute-btn:disabled {
  background-color: var(--bg-disabled);
  cursor: not-allowed;
  opacity: 0.6;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  /* 在小屏幕上，技能面板可能被隐藏或移动 */
  .skill-panel {
    display: none;
  }
}
</style>
