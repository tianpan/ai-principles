<!--
  GlobalDialog - 全局确认对话框容器

  @deprecated 此组件已废弃，请使用 Element Plus 的 ElMessageBox
  迁移指南：
  - useConfirm() composable 现在内部使用 ElMessageBox
  - 不再需要手动挂载 GlobalDialog 组件
  - App.vue 中已移除此组件的引用

  此组件已不可用，请在代码中移除对它的引用。
  新的 useConfirm API:
  - show(options?) => Promise<boolean>
  - danger(message, title?) => Promise<boolean>
  - info(message, title?) => Promise<boolean>
-->

<template>
  <!-- 此组件已废弃，不再渲染任何内容 -->
  <div class="deprecated-global-dialog" style="display: none;"></div>
</template>

<script setup lang="ts">
// 此组件已废弃
// 请使用 useConfirm() composable，它现在内部使用 Element Plus ElMessageBox
</script>

<style scoped>
/* 此组件已废弃 */
</style>
