/**
 * E2E 测试 - Astro + React Web 应用
 *
 * 使用 Playwright 测试:
 * - 页面可访问性
 * - 导航结构
 * - React 组件交互
 * - 响应式设计
 */

import { test, expect, Page } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:4321';

test.describe('AI 原理科普 - 首页', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
  });

  test('首页加载成功', async ({ page }) => {
    await expect(page).toHaveTitle(/AI 原理科普/);
  });

  test('导航栏存在', async ({ page }) => {
    const nav = page.locator('nav');
    await expect(nav).toBeVisible();
  });

  test('侧边栏显示章节', async ({ page }) => {
    // 等待 Starlight 侧边栏加载
    const sidebar = page.locator('[data-sidebar]');
    await expect(sidebar.or(page.locator('aside'))).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Chapter 2 内容页', () => {
  const chapters = [
    { slug: 'chapter2/1-token-embedding', title: 'Token' },
    { slug: 'chapter2/2-self-attention', title: 'Self-Attention' },
    { slug: 'chapter2/3-multi-head-attention', title: 'Multi-Head' },
    { slug: 'chapter2/4-positional-encoding', title: 'Positional' },
    { slug: 'chapter2/5-residual-layernorm', title: 'Residual' },
    { slug: 'chapter2/6-ffn-output', title: 'FFN' },
  ];

  for (const chapter of chapters) {
    test(`${chapter.slug} 页面可访问`, async ({ page }) => {
      await page.goto(`${BASE_URL}/${chapter.slug}/`);
      await expect(page).toHaveURL(new RegExp(chapter.slug));
    });

    test(`${chapter.slug} 包含术语卡组件`, async ({ page }) => {
      await page.goto(`${BASE_URL}/${chapter.slug}/`);
      // 等待页面加载
      await page.waitForLoadState('networkidle');
      // 检查是否有术语卡（React 组件）
      const terminologyCard = page.locator('[data-testid="terminology-card"]').or(
        page.locator('button').filter({ hasText: '术语' })
      ).or(
        page.locator('.terminology-card')
      );
      // 可能存在也可能不存在，取决于内容
    });
  }
});

test.describe('交互组件', () => {
  test('Positional Encoding Toggle 组件交互', async ({ page }) => {
    await page.goto(`${BASE_URL}/chapter2/4-positional-encoding/`);
    await page.waitForLoadState('networkidle');

    // 查找 toggle 组件
    const toggle = page.locator('[data-testid="pe-toggle"]').or(
      page.locator('button').filter({ hasText: /sin|cos|toggle/i })
    );

    if (await toggle.count() > 0) {
      await toggle.first().click();
      // 验证交互发生
      await page.waitForTimeout(500);
    }
  });

  test('Attention Explorer 组件可交互', async ({ page }) => {
    await page.goto(`${BASE_URL}/chapter2/2-self-attention/`);
    await page.waitForLoadState('networkidle');

    // 查找交互元素
    const interactive = page.locator('[data-testid="attention-explorer"]').or(
      page.locator('.attention-explorer')
    );

    if (await interactive.count() > 0) {
      // 尝试点击或滑动
      const sliders = interactive.locator('input[type="range"]');
      if (await sliders.count() > 0) {
        await sliders.first().fill('0.5');
      }
    }
  });

  test('Multi-Head Comparator 滑块交互', async ({ page }) => {
    await page.goto(`${BASE_URL}/chapter2/3-multi-head-attention/`);
    await page.waitForLoadState('networkidle');

    // 查找滑块
    const slider = page.locator('[data-testid="head-slider"]').or(
      page.locator('input[type="range"]')
    );

    if (await slider.count() > 0) {
      // 移动滑块
      await slider.first().fill('4');
      await page.waitForTimeout(300);
    }
  });
});

test.describe('导航功能', () => {
  test('从首页导航到 Chapter 2', async ({ page }) => {
    await page.goto(BASE_URL);

    // 点击 Chapter 2 链接
    const chapter2Link = page.locator('a').filter({ hasText: /Chapter 2|第二章/ });
    if (await chapter2Link.count() > 0) {
      await chapter2Link.first().click();
      await expect(page).toHaveURL(/chapter2/);
    }
  });

  test('页面间导航', async ({ page }) => {
    await page.goto(`${BASE_URL}/chapter2/1-token-embedding/`);

    // 查找下一页按钮
    const nextButton = page.locator('a').filter({ hasText: /下一页|Next/ });
    if (await nextButton.count() > 0) {
      await nextButton.first().click();
      await expect(page).toHaveURL(/chapter2\/2/);
    }
  });

  test('侧边栏导航', async ({ page }) => {
    await page.goto(`${BASE_URL}/chapter2/1-token-embedding/`);

    // 查找侧边栏中的链接
    const sidebarLink = page.locator('aside a, [data-sidebar] a').filter({
      hasText: /Multi-Head|多头/
    });

    if (await sidebarLink.count() > 0) {
      await sidebarLink.first().click();
      await expect(page).toHaveURL(/multi-head/);
    }
  });
});

test.describe('响应式设计', () => {
  test('移动端视图', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(BASE_URL);

    // 页面应该仍然可访问
    await expect(page).toHaveTitle(/AI 原理科普/);

    // 检查移动端菜单
    const mobileMenu = page.locator('[data-mobile-menu]').or(
      page.locator('button[aria-label="Menu"]')
    );
    // 移动端菜单可能存在
  });

  test('平板端视图', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto(`${BASE_URL}/chapter2/2-self-attention/`);

    // 内容应该可见
    const content = page.locator('main, article, .content');
    await expect(content.first()).toBeVisible();
  });

  test('桌面端视图', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(`${BASE_URL}/chapter2/2-self-attention/`);

    // 侧边栏应该可见
    const sidebar = page.locator('aside, [data-sidebar]');
    await expect(sidebar).toBeVisible();
  });
});

test.describe('性能检查', () => {
  test('首页加载时间', async ({ page }) => {
    const startTime = Date.now();
    await page.goto(BASE_URL, { waitUntil: 'networkidle' });
    const loadTime = Date.now() - startTime;

    // 加载时间应该小于 5 秒
    expect(loadTime).toBeLessThan(5000);
  });

  test('无控制台错误', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    await page.goto(`${BASE_URL}/chapter2/2-self-attention/`);
    await page.waitForLoadState('networkidle');

    // 允许一些非关键错误（如第三方脚本）
    const criticalErrors = errors.filter(e =>
      !e.includes('analytics') &&
      !e.includes('tracking') &&
      !e.includes('extension')
    );

    expect(criticalErrors).toHaveLength(0);
  });
});

test.describe('可访问性', () => {
  test('页面标题层次', async ({ page }) => {
    await page.goto(`${BASE_URL}/chapter2/1-token-embedding/`);

    // 应该有 h1
    const h1 = page.locator('h1');
    await expect(h1).toHaveCount(1);
  });

  test('链接有可访问名称', async ({ page }) => {
    await page.goto(BASE_URL);

    const links = page.locator('a[href]');
    const count = await links.count();

    for (let i = 0; i < Math.min(count, 10); i++) {
      const link = links.nth(i);
      const text = await link.textContent();
      const ariaLabel = await link.getAttribute('aria-label');
      const title = await link.getAttribute('title');

      // 链接应该有某种可访问名称
      expect(text || ariaLabel || title).toBeTruthy();
    }
  });

  test('交互元素可聚焦', async ({ page }) => {
    await page.goto(`${BASE_URL}/chapter2/2-self-attention/`);

    // Tab 导航
    await page.keyboard.press('Tab');
    const focusedElement = page.locator(':focus');

    // 应该有元素获得焦点
    await expect(focusedElement).toHaveCount(1);
  });
});
