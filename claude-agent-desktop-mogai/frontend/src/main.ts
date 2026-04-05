/**
 * Towngas Manus Frontend - 应用入口
 *
 * 这是Vue3应用的入口文件，负责：
 * 1. 创建Vue应用实例
 * 2. 挂载根组件
 * 3. 配置全局设置
 * 4. 配置 Element Plus 组件库
 */

import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import 'element-plus/dist/index.css'

import App from './App.vue'
import './styles/main.css'

// 创建Vue应用
const app = createApp(App)

// 注册 Element Plus
app.use(ElementPlus, { size: 'default', zIndex: 3000 })

// 注册所有 Element Plus 图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 挂载到#app元素
app.mount('#app')

// 开发环境下输出提示
if (import.meta.env.DEV) {
  console.log('Towngas Manus Frontend - Development Mode')
  console.log('API Base URL:', import.meta.env.VITE_API_BASE_URL || '/api')
}
