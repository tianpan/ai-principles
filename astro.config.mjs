import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  integrations: [
    starlight({
      title: 'AI 原理科普',
      defaultLocale: 'root',
      locales: {
        root: {
          label: '简体中文',
          lang: 'zh-CN',
        },
      },
      sidebar: [
        {
          label: '开始学习',
          items: [
            { label: '欢迎', slug: '' },
            { label: '预备知识', slug: 'prerequisites' },
          ],
        },
        {
          label: 'Chapter 2: Transformer 核心',
          autogenerate: { directory: 'chapter2' },
        },
        {
          label: 'Mini Transformer Lab',
          items: [
            { label: '交互实验室', slug: 'lab/playground' },
            { label: '实验室介绍', slug: 'lab/intro' },
            { label: '代码实现', slug: 'lab/implementation' },
            { label: '训练与推理', slug: 'lab/training' },
          ],
        },
      ],
      social: {
        github: 'https://github.com/tianpan/ai-principles',
      },
    }),
    react(),
    tailwind({
      applyBaseStyles: false,
    }),
  ],
  output: 'static',
  site: 'https://tianpan.github.io',
  base: '/ai-principles',
});
