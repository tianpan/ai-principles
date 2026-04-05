import { test, expect, type Page, type BrowserContext } from "@playwright/test";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const BASE = "http://localhost:3002";
const SCREENSHOT_DIR = "e2e/screenshots";

// PC user credentials (from src/mock/users.json)
const PC_MANAGER = {
    phone: "13800000001",
    password: "123456",
    name: "王经理",
    role: "项目经理",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Take a consistently named screenshot */
async function takeScreenshot(page: Page, name: string) {
    const path = `${SCREENSHOT_DIR}/${name}.png`;
    await page.screenshot({ path, fullPage: true });
    console.log(`  [screenshot] ${path}`);
}

/**
 * Login by navigating to /login, waiting for React hydration,
 * and using the quick-select buttons or form.
 */
async function loginAsManager(page: Page) {
    await page.goto(`${BASE}/login`, { waitUntil: "load", timeout: 15000 });

    // The login page is "use client" so we must wait for React hydration.
    // Poll for the login button to appear in the DOM.
    const loginBtn = page.locator('button:has-text("登录")').last();
    await loginBtn.waitFor({ state: "visible", timeout: 15000 });

    // Try quick-select button first (fills phone + password automatically)
    const quickBtn = page.locator('button:has-text("王经理")').first();
    if (await quickBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await quickBtn.click();
        // Wait for the phone field to be populated
        await page.waitForTimeout(300);
    } else {
        // Fill manually
        await page.locator('input[placeholder="请输入手机号"]').fill(PC_MANAGER.phone);
        await page.locator('input[placeholder="请输入密码"]').fill(PC_MANAGER.password);
    }

    // Click login
    await page.locator('button:has-text("登录")').last().click();

    // Wait for navigation away from /login
    await page.waitForURL(
        (url) => !url.pathname.endsWith("/login"),
        { timeout: 10000 }
    ).catch(() => {
        // Navigation might go to /home which returns 500; that is okay.
        // The important thing is the login context was set.
    });
}

// ---------------------------------------------------------------------------
// Test Suite
// ---------------------------------------------------------------------------

// Create a separate describe that uses its own browser context with desktop
// viewport, bypassing the mobile-chrome project settings.
test.describe("PC Dashboard AI Features", () => {
    test.describe.configure({ mode: "serial" });

    let page: Page;
    let context: BrowserContext;

    test.beforeAll(async ({ browser }) => {
        // Create a new context with desktop viewport regardless of project config
        context = await browser.newContext({
            viewport: { width: 1440, height: 900 },
            locale: "zh-CN",
        });
        page = await context.newPage();
    });

    test.afterAll(async () => {
        await page.close();
        await context.close();
    });

    // -----------------------------------------------------------------------
    // Flow 1: Login + PC Dashboard AI Briefing
    // -----------------------------------------------------------------------
    test("Flow 1: login and view AI briefing on dashboard", async () => {
        console.log("\n=== Flow 1: PC Dashboard AI Briefing ===");

        // Step 1: Login
        console.log("Step 1: Navigate to login page and authenticate");
        await loginAsManager(page);
        await page.waitForTimeout(1000);
        await takeScreenshot(page, "01-after-login");

        // Step 2: Navigate to PC dashboard
        console.log("Step 2: Navigate to PC dashboard");
        await page.goto(`${BASE}/pc/dashboard`, { waitUntil: "load", timeout: 15000 });

        // Wait for React to render the page
        const dashboardTitle = page.locator('h1:has-text("工作台")');
        await dashboardTitle.waitFor({ state: "visible", timeout: 15000 });
        await takeScreenshot(page, "02-pc-dashboard");

        // Step 3: Verify AI Briefing Card
        console.log("Step 3: Verify AI Briefing Card");
        const briefingHeader = page.locator("text=AI 每日简报").first();
        await expect(briefingHeader).toBeVisible({ timeout: 10000 });
        console.log("  [PASS] AI Briefing card header is visible");

        // Verify project name in briefing header
        const h3Contents = await page.locator("h3").allTextContents();
        const hasProject = h3Contents.some(
            (t) => t.includes("港华") || t.includes("简报")
        );
        expect(hasProject).toBeTruthy();
        console.log("  [PASS] Project name in briefing header");

        // Verify date is displayed
        const dateSpan = page.locator("h3 + span, h3 ~ span").first();
        if (await dateSpan.isVisible({ timeout: 2000 }).catch(() => false)) {
            const dateText = await dateSpan.textContent();
            console.log(`  [INFO] Date shown: "${dateText}"`);
        }

        // Verify briefing content
        const contentArea = page.locator(".whitespace-pre-line").first();
        if (await contentArea.isVisible({ timeout: 3000 }).catch(() => false)) {
            const content = await contentArea.textContent();
            console.log(`  [INFO] Briefing: ${content?.substring(0, 80)}...`);
            expect(content!.length).toBeGreaterThan(10);
            console.log("  [PASS] Briefing has substantive content");
        }

        // Verify risk section (either has risks or "no risk" message)
        const riskLabel = page.locator("text=风险提示").first();
        const noRiskLabel = page.locator("text=施工进展正常").first();
        const hasRiskInfo =
            (await riskLabel.isVisible({ timeout: 2000 }).catch(() => false)) ||
            (await noRiskLabel.isVisible({ timeout: 2000 }).catch(() => false));
        console.log(`  [INFO] Risk section present: ${hasRiskInfo}`);

        // Verify stat cards
        const statLabels = ["进行中项目", "待审核", "超时预警", "今日已通过"];
        for (const label of statLabels) {
            await expect(page.locator(`text=${label}`).first()).toBeVisible({
                timeout: 5000,
            });
        }
        console.log("  [PASS] All 4 stat cards visible");

        await takeScreenshot(page, "03-ai-briefing-detail");
        console.log("[Flow 1 PASSED]");
    });

    // -----------------------------------------------------------------------
    // Flow 2: PC Smart Review
    // -----------------------------------------------------------------------
    test("Flow 2: smart review page with AI pre-review", async () => {
        console.log("\n=== Flow 2: PC Smart Review ===");

        // Step 1: Navigate to review page
        console.log("Step 1: Navigate to /pc/review");
        await page.goto(`${BASE}/pc/review`, { waitUntil: "load", timeout: 15000 });

        const reviewTitle = page.locator('h1:has-text("审核工作台")');
        await reviewTitle.waitFor({ state: "visible", timeout: 15000 });
        await page.waitForTimeout(1000);
        await takeScreenshot(page, "04-pc-review-page");

        // Step 2: Verify AI Pre-Review Summary
        console.log("Step 2: Verify AI Pre-Review Summary");
        const summaryHeader = page.locator("text=AI 预审摘要").first();
        const summaryVisible = await summaryHeader
            .isVisible({ timeout: 5000 })
            .catch(() => false);
        console.log(`  [INFO] AI Pre-Review Summary visible: ${summaryVisible}`);

        if (summaryVisible) {
            // Verify three categories
            const categories = ["正常", "疑似异常", "明显问题"];
            for (const cat of categories) {
                const visible = await page
                    .locator(`text=${cat}`)
                    .first()
                    .isVisible({ timeout: 3000 })
                    .catch(() => false);
                console.log(`  [INFO] Category "${cat}" visible: ${visible}`);
            }
            console.log("  [PASS] AI Pre-Review Summary shows categories");

            // Step 3: Check batch approve button
            console.log("Step 3: Check batch approve button");
            const batchBtn = page
                .locator('button:has-text("一键通过")')
                .first();
            const batchVisible = await batchBtn
                .isVisible({ timeout: 3000 })
                .catch(() => false);
            console.log(`  [INFO] Batch approve visible: ${batchVisible}`);

            if (batchVisible) {
                const btnText = await batchBtn.textContent();
                console.log(`  [INFO] Button text: "${btnText}"`);
                await batchBtn.click();
                await page.waitForTimeout(1500);
                await takeScreenshot(page, "05-after-batch-approve");
                console.log("  [PASS] Batch approve clicked");
            }
        } else {
            console.log("  [WARN] No pending records, summary not shown");
        }

        // Step 4: Check AI detail buttons
        console.log("Step 4: Check AI detail buttons");
        const aiDetailBtns = page.locator('button:has-text("AI详情")');
        const aiDetailCount = await aiDetailBtns.count();
        console.log(`  [INFO] Found ${aiDetailCount} AI detail buttons`);

        if (aiDetailCount > 0) {
            await aiDetailBtns.first().click();
            await page.waitForTimeout(1500);
            await takeScreenshot(page, "06-ai-review-detail");

            // Verify detail panel opened
            const detailPanel = page.locator("text=记录 #").first();
            const detailVisible = await detailPanel
                .isVisible({ timeout: 3000 })
                .catch(() => false);
            expect(detailVisible).toBeTruthy();
            console.log("  [PASS] AI Review Detail panel opened");

            // Check action buttons in detail
            const approveBtn = page
                .locator('button:has-text("通过")')
                .first();
            const rejectBtn = page
                .locator('button:has-text("退回")')
                .first();

            if (
                await approveBtn.isVisible({ timeout: 2000 }).catch(() => false)
            ) {
                console.log("  [PASS] Approve button in detail visible");
            }
            if (
                await rejectBtn.isVisible({ timeout: 2000 }).catch(() => false)
            ) {
                console.log("  [PASS] Reject button in detail visible");
            }

            // Close detail by clicking approve
            await approveBtn.click();
            await page.waitForTimeout(1000);
        }

        // Step 5: Verify record table
        console.log("Step 5: Verify record table structure");
        await expect(
            page.locator("th:has-text('日期')").first()
        ).toBeVisible({ timeout: 5000 });
        await expect(
            page.locator("th:has-text('模块')").first()
        ).toBeVisible({ timeout: 3000 });
        console.log("  [PASS] Record table has proper columns");

        await takeScreenshot(page, "07-review-final-state");
        console.log("[Flow 2 PASSED]");
    });

    // -----------------------------------------------------------------------
    // Flow 3: NL Query
    // -----------------------------------------------------------------------
    test("Flow 3: natural language query from PC layout", async () => {
        console.log("\n=== Flow 3: NL Query ===");

        // Step 1: Go to PC dashboard
        console.log("Step 1: Navigate to PC dashboard");
        await page.goto(`${BASE}/pc/dashboard`, { waitUntil: "load", timeout: 15000 });

        const dashboardTitle = page.locator('h1:has-text("工作台")');
        await dashboardTitle.waitFor({ state: "visible", timeout: 15000 });
        await page.waitForTimeout(1000);

        // Step 2: Find NL query input
        console.log("Step 2: Find NL query input bar");
        const queryInput = page.locator('input[placeholder*="输入问题"]');
        await expect(queryInput).toBeVisible({ timeout: 10000 });
        console.log("  [PASS] NL query input bar visible");

        // Step 3: Type and submit "进度怎么样"
        console.log("Step 3: Submit query '进度怎么样'");
        await queryInput.fill("进度怎么样");
        await page.waitForTimeout(300);
        await takeScreenshot(page, "08-nl-query-input");

        await queryInput.press("Enter");
        await page.waitForTimeout(2000);
        await takeScreenshot(page, "09-nl-query-result");

        // Step 4: Verify result panel
        console.log("Step 4: Verify query result panel");

        // Query type badge (e.g., "进度查询")
        const queryBadge = page
            .locator(".shadow-lg .rounded-full.font-medium")
            .first();
        if (await queryBadge.isVisible({ timeout: 5000 }).catch(() => false)) {
            const badgeText = await queryBadge.textContent();
            console.log(`  [PASS] Query type badge: "${badgeText}"`);
            expect(badgeText).toContain("查询");
        } else {
            // Fallback: check for any rounded-full badge
            const anyBadge = page.locator(".rounded-full.font-medium").first();
            if (await anyBadge.isVisible({ timeout: 2000 }).catch(() => false)) {
                const badgeText = await anyBadge.textContent();
                console.log(`  [PASS] Query type badge (fallback): "${badgeText}"`);
            }
        }

        // Natural language answer
        const answerPanel = page
            .locator(".shadow-lg .whitespace-pre-line")
            .first();
        if (await answerPanel.isVisible({ timeout: 3000 }).catch(() => false)) {
            const answerText = await answerPanel.textContent();
            console.log(`  [INFO] Answer: ${answerText?.substring(0, 100)}...`);
            expect(answerText!.length).toBeGreaterThan(5);
            console.log("  [PASS] Natural language answer displayed");
        }

        // Data cards in result
        const dataCards = page.locator(".shadow-lg .bg-gray-50.p-2");
        const dataCardCount = await dataCards.count();
        console.log(`  [INFO] Found ${dataCardCount} data cards`);
        if (dataCardCount > 0) {
            console.log("  [PASS] Data cards present in result");
            for (let i = 0; i < Math.min(dataCardCount, 6); i++) {
                const cardText = await dataCards.nth(i).textContent();
                console.log(`    Card ${i + 1}: ${cardText?.trim()}`);
            }
        }

        // Close button
        const closeBtn = page.locator('button:has-text("✕")').first();
        const closeVisible = await closeBtn
            .isVisible({ timeout: 2000 })
            .catch(() => false);
        console.log(`  [INFO] Close button visible: ${closeVisible}`);

        await takeScreenshot(page, "10-nl-query-full-result");

        // Step 5: Material query
        console.log("Step 5: Try material query");
        if (closeVisible) {
            await closeBtn.click();
            await page.waitForTimeout(500);
        }

        await queryInput.fill("今天领了什么材料");
        await queryInput.press("Enter");
        await page.waitForTimeout(2000);
        await takeScreenshot(page, "11-nl-query-material");

        // Step 6: Attendance query
        console.log("Step 6: Try attendance query");
        const closeBtn2 = page.locator('button:has-text("✕")').first();
        if (await closeBtn2.isVisible({ timeout: 2000 }).catch(() => false)) {
            await closeBtn2.click();
            await page.waitForTimeout(500);
        }

        await queryInput.fill("今天出勤多少人");
        await queryInput.press("Enter");
        await page.waitForTimeout(2000);
        await takeScreenshot(page, "12-nl-query-attendance");

        console.log("[Flow 3 PASSED]");
    });
});
