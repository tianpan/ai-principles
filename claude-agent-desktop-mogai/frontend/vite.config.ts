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
    rollupOptions: {
      output: {
        // 分包策略
        manualChunks: {
          vue: ['vue'],
          markdown: ['marked', 'highlight.js', 'dompurify'],
        },
      },
    },
  },
})
