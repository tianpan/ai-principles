import { defineConfig } from "@playwright/test";

export default defineConfig({
    testDir: "./e2e",
    fullyParallel: false,
    forbidOnly: !!process.env.CI,
    retries: process.env.CI ? 2 : 0,
    workers: 1,
    reporter: [
        ["html", { outputFolder: "e2e/report" }],
        ["list"],
    ],
    use: {
        baseURL: "http://localhost:3002",
        trace: "on-first-retry",
        screenshot: "on",
        video: "retain-on-failure",
        actionTimeout: 10000,
        locale: "zh-CN",
    },
    projects: [
        {
            name: "mobile-chrome",
            use: {
                viewport: { width: 375, height: 812 },
                userAgent:
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
                isMobile: true,
                hasTouch: true,
            },
        },
    ],
    expect: {
        timeout: 10000,
    },
});
