<!--
  Towngas Manus Frontend - 技能面板组件

  功能：
  1. 显示技能列表
  2. 技能分类显示
  3. 技能执行按钮
  4. 技能状态显示

  已迁移到 Element Plus：
  - el-button circle 替代刷新按钮
  - el-collapse 替代技能分组
  - el-card shadow="hover" 替代技能卡片
  - el-tag 替代状态标签
  - el-button type="primary" 替代执行按钮
-->

<template>
  <div class="skill-panel">
    <!-- 头部 -->
    <div class="skill-header">
      <h2 class="skill-title">技能面板</h2>
      <el-button
        circle
        @click="handleRefresh"
        title="刷新技能列表"
      >
        <el-icon><Refresh /></el-icon>
      </el-button>
    </div>

    <!-- 技能列表 -->
    <div class="skill-list">
      <!-- Element Plus 空状态 -->
      <el-empty
        v-if="skills.length === 0"
        :image-size="80"
        description="暂无可用技能"
      >
        <template #image>
          <el-icon :size="48" color="var(--text-muted)">
            <Lightning />
          </el-icon>
        </template>
      </el-empty>

      <!-- Element Plus 折叠面板显示技能分组 -->
      <el-collapse v-else v-model="activeCollapse" class="skill-collapse">
        <el-collapse-item
          v-for="group in groupedSkills"
          :key="group.category"
          :name="group.category"
        >
          <template #title>
            <div class="group-title">
              {{ group.category || '其他' }}
              <el-tag size="small" type="info" class="group-count">
                {{ group.skills.length }}
              </el-tag>
            </div>
          </template>

          <!-- Element Plus 卡片显示技能 -->
          <el-card
            v-for="skill in group.skills"
            :key="skill.id"
            shadow="hover"
            class="skill-card"
          >
            <!-- 技能图标和名称 -->
            <div class="skill-main">
              <span class="skill-icon">
                <el-icon :size="16">
                  <component :is="getSkillIcon(skill.id)" />
                </el-icon>
              </span>
              <div class="skill-info">
                <div class="skill-name">{{ skill.name }}</div>
                <div class="skill-desc">{{ skill.description }}</div>
              </div>
            </div>

            <!-- Element Plus 状态标签 -->
            <div class="skill-footer">
              <el-tag :type="getStatusTagType(skill.status)" size="small">
                {{ getStatusText(skill.status) }}
              </el-tag>

              <!-- Element Plus 执行按钮 -->
              <el-button
                type="primary"
                size="small"
                :disabled="skill.status !== 'available'"
                @click="handleExecute(skill.id)"
              >
                <el-icon class="el-icon--left"><VideoPlay /></el-icon>
                执行
              </el-button>
            </div>
          </el-card>
        </el-collapse-item>
      </el-collapse>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 技能面板组件逻辑
 * 已迁移到 Element Plus 组件库
 */

import { ref, computed } from 'vue'
// Element Plus 图标
import {
  Refresh,
  Lightning,
  VideoPlay,
  Clock,
  Calendar,
  Coin,
  Document,
  DataLine
} from '@element-plus/icons-vue'
import type { Skill, SkillStatus } from '@/types'
import type { Component } from 'vue'

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

// ==================== 状态 ====================

// 折叠面板激活项
const activeCollapse = ref<string[]>([])

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
  const result = Array.from(groups.entries()).map(([category, skills]) => ({
    category,
    skills,
  }))

  // 默认展开所有分类
  if (activeCollapse.value.length === 0) {
    activeCollapse.value = result.map(g => g.category)
  }

  return result
})

// ==================== 方法 ====================

/**
 * 获取技能图标
 *
 * @param skillId 技能ID
 * @returns 图标组件
 */
const getSkillIcon = (skillId: string): Component => {
  const iconMap: Record<string, Component> = {
    'get_current_time': Clock,
    'query_station': Coin,
    'generate_report': Document,
    'check_device': DataLine,
    'schedule_task': Calendar,
  }
  return iconMap[skillId] || Lightning
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
 * 获取 Element Plus Tag 类型
 *
 * @param status 技能状态
 * @returns Tag 类型
 */
const getStatusTagType = (status: SkillStatus): 'success' | 'warning' | 'danger' | 'info' => {
  const typeMap: Record<SkillStatus, 'success' | 'warning' | 'danger' | 'info'> = {
    available: 'success',
    running: 'warning',
    error: 'danger',
    disabled: 'info',
  }
  return typeMap[status] || 'info'
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

/* 技能列表 */
.skill-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

/* Element Plus 折叠面板样式覆盖 */
.skill-collapse {
  border: none;
}

.skill-collapse :deep(.el-collapse-item__header) {
  background-color: transparent;
  border-bottom: none;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  height: auto;
  padding: 8px 12px;
}

.skill-collapse :deep(.el-collapse-item__wrap) {
  border-bottom: none;
  background-color: transparent;
}

.skill-collapse :deep(.el-collapse-item__content) {
  padding-bottom: 8px;
}

/* 分组标题 */
.group-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.group-count {
  font-weight: 400;
  font-size: 0.625rem;
}

/* Element Plus 卡片样式覆盖 */
.skill-card {
  margin-bottom: 8px;
  border-radius: 8px;
}

.skill-card :deep(.el-card__body) {
  padding: 12px;
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

/* 技能底部 */
.skill-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  /* 在小屏幕上，技能面板可能被隐藏或移动 */
  .skill-panel {
    display: none;
  }
}
</style>
