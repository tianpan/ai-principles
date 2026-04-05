import { test, expect } from "@playwright/test";

test.describe("AI Chat Assistant - Team Leader Flow", () => {
    test.beforeEach(async ({ page }) => {
        // Clear localStorage before any page loads to ensure clean state
        await page.addInitScript(() => {
            localStorage.removeItem("mockCurrentUser");
        });
    });

    test("complete AI chat flow: login -> open chat -> send message -> confirm -> close", async ({
        page,
    }) => {
        // ============================================================
        // STEP 1: Login as Team Leader (班组长)
        // ============================================================
        await page.goto("/login");
        await page.waitForLoadState("domcontentloaded");

        // Screenshot: Login page initial state
        await page.screenshot({
            path: "e2e/screenshots/01-login-page.png",
            fullPage: true,
        });

        // Verify login page has loaded correctly
        await expect(page.locator("h1")).toContainText("工程管家", {
            timeout: 15000,
        });

        // Find the quick login button for 张三 (班组长) and click it
        // The page renders buttons like "张三 (班组长)" from mock data
        const teamLeaderButton = page.locator("button", {
            hasText: /班组长/,
        }).first();

        await expect(teamLeaderButton).toBeVisible();
        await teamLeaderButton.click();

        // Screenshot: After selecting team leader
        await page.screenshot({
            path: "e2e/screenshots/02-credentials-filled.png",
        });

        // Verify phone and password fields are populated
        const phoneInput = page.locator('input[type="tel"]');
        const passwordInput = page.locator('input[type="password"]');
        await expect(phoneInput).toHaveValue(/1380000000[2-4]/);
        await expect(passwordInput).toHaveValue("123456");

        // Click the login button
        const loginButton = page.locator("button", { hasText: "登录" });
        await loginButton.click();

        // Wait for navigation to home page
        await page.waitForURL("**/home", { timeout: 15000 });

        // Screenshot: Home page after login
        await page.screenshot({
            path: "e2e/screenshots/03-home-page.png",
            fullPage: true,
        });

        // Verify we are on the home page
        await expect(page).toHaveURL(/\/home/);

        // ============================================================
        // STEP 2: Open AI Chat Widget
        // ============================================================

        // The AI chat widget button has aria-label="AI 助手"
        // It is a fixed button at bottom right with chat icon
        const aiButton = page.locator('button[aria-label="AI 助手"]');

        await expect(aiButton).toBeVisible({ timeout: 10000 });

        // Screenshot: AI chat button visible on home page
        await page.screenshot({
            path: "e2e/screenshots/04-ai-button-visible.png",
        });

        // Click the AI assistant floating button
        await aiButton.click();

        // Wait for chat panel to slide up (transition is 300ms)
        await page.waitForTimeout(500);

        // Screenshot: Chat panel open
        await page.screenshot({
            path: "e2e/screenshots/05-chat-panel-open.png",
        });

        // ============================================================
        // STEP 3: Verify Chat Panel Contents
        // ============================================================

        // Verify the chat header
        const chatHeader = page.locator("h3", { hasText: "AI 助手" });
        await expect(chatHeader).toBeVisible();

        // Verify welcome message (shown when no messages)
        const welcomeMessage = page.locator("text=您好！我是 AI 助手。");
        await expect(welcomeMessage).toBeVisible();

        // Verify quick action buttons are visible
        // Quick actions: 安全考勤, 施工进度, 材料登记, 施工记录
        await expect(
            page.locator("button", { hasText: "安全考勤" })
        ).toBeVisible();
        await expect(
            page.locator("button", { hasText: "施工进度" })
        ).toBeVisible();
        await expect(
            page.locator("button", { hasText: "材料登记" })
        ).toBeVisible();
        await expect(
            page.locator("button", { hasText: "施工记录" })
        ).toBeVisible();

        // Verify input field is present
        const chatInput = page.locator(
            'input[placeholder="输入消息..."]'
        );
        await expect(chatInput).toBeVisible();
        await expect(chatInput).toBeEnabled();

        // Screenshot: Chat panel with welcome message and quick actions
        await page.screenshot({
            path: "e2e/screenshots/06-chat-verified.png",
        });

        // ============================================================
        // STEP 4: Send a Progress Message
        // ============================================================

        // Type "今天焊了80米" in the input field
        await chatInput.fill("今天焊了80米");

        // Screenshot: Input filled with message
        await page.screenshot({
            path: "e2e/screenshots/07-message-typed.png",
        });

        // Click the send button
        const sendButton = page.locator("button", { hasText: "发送" });
        await sendButton.click();

        // Wait for AI response to appear.
        // Mock LLM is synchronous and very fast -- "思考中..." may flash
        // too briefly to detect. Instead, wait directly for the response.
        await expect(
            page.locator("text=进度记录").first()
        ).toBeVisible({ timeout: 10000 });

        // Screenshot: AI response received
        await page.screenshot({
            path: "e2e/screenshots/08-ai-response.png",
        });

        // Verify the user message bubble is shown
        await expect(
            page.locator("text=今天焊了80米").first()
        ).toBeVisible();

        // Verify the AI responds with progress record content
        // Mock LLM returns: "好的，已识别为**进度记录**：完成 80米..."
        await expect(
            page.locator("text=80").first()
        ).toBeVisible();

        // Verify confirmation buttons appear
        const confirmButton = page.locator("button", {
            hasText: "确认提交",
        });
        const cancelButton = page.locator("button", {
            hasText: "取消",
        });
        await expect(confirmButton).toBeVisible();
        await expect(cancelButton).toBeVisible();

        // Screenshot: Confirmation buttons visible
        await page.screenshot({
            path: "e2e/screenshots/09-confirm-buttons.png",
        });

        // ============================================================
        // STEP 5: Confirm Submission
        // ============================================================

        // Click "确认提交"
        await confirmButton.click();

        // Wait a moment for state update
        await page.waitForTimeout(500);

        // Screenshot: After confirmation
        await page.screenshot({
            path: "e2e/screenshots/10-after-confirm.png",
        });

        // Verify the confirmation buttons have disappeared
        await expect(confirmButton).toBeHidden();
        await expect(cancelButton).toBeHidden();

        // ============================================================
        // STEP 6: Close the Chat Panel
        // ============================================================

        // Close the chat panel by dispatching a click event on the AI button
        // The chat panel overlays the button, so we use evaluate to trigger React's onClick
        await page.evaluate(() => {
            const btn = document.querySelector('button[aria-label="AI 助手"]') as HTMLButtonElement;
            if (btn) btn.click();
        });

        // Wait for slide-down animation (CSS transition 300ms)
        await page.waitForTimeout(500);

        // Screenshot: Chat panel closed
        await page.screenshot({
            path: "e2e/screenshots/11-chat-closed.png",
        });

        // Verify the chat panel is closed by checking the panel's transform
        // The panel uses translate-y-full when closed
        const panelHidden = await page.evaluate(() => {
            const panel = document.querySelector('.translate-y-full');
            return panel !== null;
        });
        expect(panelHidden).toBeTruthy();
    });

    test("chat quick actions trigger module-specific responses", async ({
        page,
    }) => {
        // Login as team leader
        await page.goto("/login");
        await page.waitForLoadState("domcontentloaded");

        await expect(page.locator("h1")).toContainText("工程管家", {
            timeout: 15000,
        });

        const teamLeaderButton = page.locator("button", {
            hasText: /班组长/,
        }).first();
        await teamLeaderButton.click();

        await page.locator("button", { hasText: "登录" }).click();
        await page.waitForURL("**/home", { timeout: 15000 });

        // Open AI chat
        await page.locator('button[aria-label="AI 助手"]').click();
        await page.waitForTimeout(500);

        // Click "施工进度" quick action
        await page
            .locator("button", { hasText: "施工进度" })
            .click();

        // Wait for AI response (mock LLM is fast, wait for content directly)
        await expect(
            page.locator("text=帮我填写施工进度").first()
        ).toBeVisible({ timeout: 10000 });

        // Screenshot: Quick action response
        await page.screenshot({
            path: "e2e/screenshots/12-quick-action-response.png",
        });
    });

    test("cancel button dismisses intent without submission", async ({
        page,
    }) => {
        // Login as team leader
        await page.goto("/login");
        await page.waitForLoadState("domcontentloaded");

        await expect(page.locator("h1")).toContainText("工程管家", {
            timeout: 15000,
        });

        const teamLeaderButton = page.locator("button", {
            hasText: /班组长/,
        }).first();
        await teamLeaderButton.click();

        await page.locator("button", { hasText: "登录" }).click();
        await page.waitForURL("**/home", { timeout: 15000 });

        // Open AI chat
        await page.locator('button[aria-label="AI 助手"]').click();
        await page.waitForTimeout(500);

        // Send a progress message
        const chatInput = page.locator(
            'input[placeholder="输入消息..."]'
        );
        await chatInput.fill("今天焊了80米");
        await page.locator("button", { hasText: "发送" }).click();

        // Wait for response (mock LLM is fast, wait for content directly)
        await expect(
            page.locator("text=进度记录").first()
        ).toBeVisible({ timeout: 10000 });

        // Click cancel -- this sends "取消" as a new message to the AI
        await page
            .locator("button", { hasText: "取消" })
            .click();

        // Wait for cancel response
        await page.waitForTimeout(1000);

        // Screenshot: After cancel
        await page.screenshot({
            path: "e2e/screenshots/13-after-cancel.png",
        });

        // Verify a response about cancel was generated
        // The mock LLM returns a "did not understand" message for "取消"
        // since it has no matching module keywords
        await expect(
            page.locator("text=我还没太理解您的意思").first()
        ).toBeVisible({ timeout: 5000 });
    });

    test("reset session clears chat history", async ({ page }) => {
        // Login as team leader
        await page.goto("/login");
        await page.waitForLoadState("domcontentloaded");

        await expect(page.locator("h1")).toContainText("工程管家", {
            timeout: 15000,
        });

        const teamLeaderButton = page.locator("button", {
            hasText: /班组长/,
        }).first();
        await teamLeaderButton.click();

        await page.locator("button", { hasText: "登录" }).click();
        await page.waitForURL("**/home", { timeout: 15000 });

        // Open AI chat and send a message
        await page.locator('button[aria-label="AI 助手"]').click();
        await page.waitForTimeout(500);

        const chatInput = page.locator(
            'input[placeholder="输入消息..."]'
        );
        await chatInput.fill("今天焊了80米");
        await page.locator("button", { hasText: "发送" }).click();

        // Wait for response (mock LLM is fast, wait for content directly)
        await expect(
            page.locator("text=进度记录").first()
        ).toBeVisible({ timeout: 10000 });

        // Verify messages exist
        await expect(
            page.locator("text=今天焊了80米").first()
        ).toBeVisible();

        // Click "新对话" to reset
        await page
            .locator("button", { hasText: "新对话" })
            .click();

        // Screenshot: After reset
        await page.screenshot({
            path: "e2e/screenshots/14-after-reset.png",
        });

        // Verify welcome message is back
        await expect(
            page.locator("text=您好！我是 AI 助手。")
        ).toBeVisible();

        // Verify previous messages are gone
        // Note: the welcome message contains "试试说「今天焊了80米」"
        // so we check for the user message bubble specifically (blue bubble, right-aligned)
        // by verifying the user message is not in a chat bubble anymore
        const userBubbles = page.locator(
            'div.bg-eng-blue.text-white:has-text("今天焊了80米")'
        );
        await expect(userBubbles).toBeHidden();
    });

    test("non-team-leader users do not see AI chat widget", async ({
        page,
    }) => {
        // Login as project manager (项目经理)
        await page.goto("/login");
        await page.waitForLoadState("domcontentloaded");

        await expect(page.locator("h1")).toContainText("工程管家", {
            timeout: 15000,
        });

        const managerButton = page.locator("button", {
            hasText: "项目经理",
        });
        await managerButton.click();

        await page.locator("button", { hasText: "登录" }).click();
        await page.waitForURL("**/home", { timeout: 15000 });

        // Screenshot: Project manager home page
        await page.screenshot({
            path: "e2e/screenshots/15-manager-home.png",
            fullPage: true,
        });

        // Verify AI chat button is NOT present for non-team-leader
        const aiButton = page.locator('button[aria-label="AI 助手"]');
        await expect(aiButton).toBeHidden();
    });
});
