/// <reference types="vite/client" />

/**
 * Vue单文件组件类型声明
 */
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<object, object, unknown>
  export default component
}

/**
 * Vite环境变量类型声明
 */
interface ImportMetaEnv {
  /** API基础URL */
  readonly VITE_API_BASE_URL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
