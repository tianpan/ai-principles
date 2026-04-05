<!--
  ToastContainer - 全局 Toast 容器

  @deprecated 此组件已废弃，请使用 Element Plus 的 ElMessage
  迁移指南：
  - useToast() composable 现在内部使用 ElMessage
  - 不再需要手动挂载 ToastContainer 组件
  - App.vue 中已移除此组件的引用

  此组件已不可用，请在代码中移除对它的引用。
  新的 useToast API:
  - success(message, duration?)
  - error(message, duration?)
  - warning(message, duration?)
  - info(message, duration?)
-->

<template>
  <!-- 此组件已废弃，不再渲染任何内容 -->
  <div class="deprecated-toast-container" style="display: none;"></div>
</template>

<script setup lang="ts">
// 此组件已废弃
// 请使用 useToast() composable，它现在内部使用 Element Plus ElMessage
</script>

<style scoped>
/* 此组件已废弃 */
</style>
