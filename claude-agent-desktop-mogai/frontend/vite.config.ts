import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      // 设置路径别名，方便导入
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    // 开发服务器配置
    port: 5173,
    host: true,
    // 代理配置，用于开发环境API请求
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // SSE流式响应代理
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/sessions': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/skills': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // 生产构建配置
    outDir: 'dist',
    sourcemap: true,
    // 提高 chunk 大小警告限制
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        // 分包策略 - 更细粒度的分包
        manualChunks: {
          // Vue 核心
          vue: ['vue'],
          // Markdown 相关库
          markdown: ['marked'],
          // 代码高亮 - 按需加载常用语言
          'highlight-core': ['highlight.js/lib/core'],
          // DOM 净化
          dompurify: ['dompurify'],
          // 图标库
          'lucide-icons': ['lucide-vue-next'],
        },
      },
    },
  },
  // 优化依赖预构建
  optimizeDeps: {
    include: ['vue', 'marked', 'dompurify', 'axios'],
  },
})
