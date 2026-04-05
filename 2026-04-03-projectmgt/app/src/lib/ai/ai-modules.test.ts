import { describe, it, expect } from "vitest";
import { generateBriefing, type BriefingInput } from "./briefing-generator";
import { executeNLQuery, type QueryDataContext } from "./nl-query";
import { preReview, generatePreReviewSummary } from "./smart-review";
import type { AuditRecord, AuditContext } from "./rule-engine";

describe("briefing-generator", () => {
    const sampleInput: BriefingInput = {
        projectId: 1,
        projectName: "XX路中压燃气管道工程",
        date: "2026-04-04",
        progressItems: [
            { name: "中压管线焊接", todayQuantity: 80, cumulativeQuantity: 620, targetQuantity: 1200, unit: "米" },
            { name: "管沟开挖", todayQuantity: 50, cumulativeQuantity: 950, targetQuantity: 1500, unit: "米" },
        ],
        materialSummary: "领：DN100镀锌钢管 30米",
        attendanceSummary: "出勤人数：5人",
        risks: ["中压管线焊接进度略有滞后"],
    };

    it("generates briefing with correct project info", () => {
        const result = generateBriefing(sampleInput);
        expect(result.projectId).toBe(1);
        expect(result.date).toBe("2026-04-04");
        expect(result.content).toContain("XX路中压燃气管道工程");
        expect(result.content).toContain("2026-04-04");
    });

    it("includes progress data in content", () => {
        const result = generateBriefing(sampleInput);
        expect(result.content).toContain("620/1200米");
        expect(result.content).toContain("52%"); // 620/1200 = 51.67% → 52%
        expect(result.content).toContain("950/1500米");
    });

    it("includes material and attendance", () => {
        const result = generateBriefing(sampleInput);
        expect(result.content).toContain("领：DN100镀锌钢管 30米");
        expect(result.content).toContain("出勤人数：5人");
    });

    it("infers risk type and severity", () => {
        const result = generateBriefing(sampleInput);
        expect(result.risks).toHaveLength(1);
        expect(result.risks[0].type).toBe("进度滞后");
        expect(result.risks[0].severity).toBe("中");
    });

    it("handles empty risks", () => {
        const input = { ...sampleInput, risks: [] };
        const result = generateBriefing(input);
        expect(result.risks).toHaveLength(0);
    });

    it("handles high severity risk", () => {
        const input = { ...sampleInput, risks: ["严重超期风险"] };
        const result = generateBriefing(input);
        expect(result.risks[0].severity).toBe("高");
    });

    it("handles zero target quantity", () => {
        const input = { ...sampleInput, progressItems: [{ name: "测试", todayQuantity: 0, cumulativeQuantity: 0, targetQuantity: 0, unit: "米" }] };
        const result = generateBriefing(input);
        expect(result.content).toContain("0%");
    });
});

describe("nl-query", () => {
    const sampleContext: QueryDataContext = {
        projectName: "XX路中压燃气管道工程",
        workItems: [
            { name: "中压管线焊接", targetQuantity: 1200, completedQuantity: 620, unit: "米" },
            { name: "管沟开挖", targetQuantity: 1500, completedQuantity: 950, unit: "米" },
        ],
        todayAttendance: 5,
        materialSummary: [
            { name: "DN100镀锌钢管", type: "领料", quantity: 30, unit: "米" },
        ],
        totalRecords: 8,
        approvedRecords: 5,
    };

    it("identifies progress query", () => {
        const result = executeNLQuery("还剩多少没焊", sampleContext);
        expect(result.queryType).toBe("进度");
        expect(result.answer).toContain("中压管线焊接");
        expect(result.data).toHaveLength(2);
    });

    it("identifies material query", () => {
        const result = executeNLQuery("今天领了多少材料", sampleContext);
        expect(result.queryType).toBe("材料");
        expect(result.answer).toContain("DN100镀锌钢管");
    });

    it("identifies attendance query", () => {
        const result = executeNLQuery("今天来了几个人", sampleContext);
        expect(result.queryType).toBe("考勤");
        expect(result.answer).toContain("5 人");
    });

    it("falls back to comprehensive query", () => {
        const result = executeNLQuery("项目情况怎么样", sampleContext);
        expect(result.queryType).toBe("综合");
        expect(result.answer).toContain("综合概况");
    });

    it("returns data cards for progress query", () => {
        const result = executeNLQuery("进度怎么样", sampleContext);
        expect(result.data.some((d) => d.label === "中压管线焊接")).toBe(true);
        expect(result.data.some((d) => d.value.includes("52%"))).toBe(true);
    });

    it("handles empty material summary", () => {
        const ctx = { ...sampleContext, materialSummary: [] };
        const result = executeNLQuery("材料领了多少", ctx);
        expect(result.answer).toContain("暂无材料记录");
    });
});

describe("smart-review", () => {
    const sampleContext: AuditContext = {
        history: [
            { date: "2026-04-03", moduleType: "进度", progress: [{ workItemId: 1, quantity: 10 }] },
            { date: "2026-04-02", moduleType: "进度", progress: [{ workItemId: 1, quantity: 10 }] },
        ],
        project: {
            id: 1,
            name: "测试项目",
            startDate: "2026-03-01",
            endDate: "2026-06-30",
            workItems: [
                { name: "焊接", targetQuantity: 1000, unit: "米", weight: 0.5 },
                { name: "开挖", targetQuantity: 500, unit: "米", weight: 0.5 },
            ],
        },
        todayAttendance: { attendeeCount: 5 },
        workers: [
            { name: "张三", position: "焊工", certifications: [{ type: "焊工证", level: "高级", expireDate: "2027-01-01" }] },
        ],
    };

    it("classifies normal records", () => {
        const records: AuditRecord[] = [
            { id: 1, projectId: 1, date: "2026-04-04", moduleType: "进度", status: "待审核", progress: [{ workItemId: 1, quantity: 10 }] },
        ];
        const result = preReview(records, sampleContext);
        expect(result.normal.length + result.suspicious.length + result.critical.length).toBe(1);
    });

    it("generates summary text", () => {
        const result = { normal: [{ id: 1, projectId: 1, date: "2026-04-04", moduleType: "进度", status: "待审核" }], suspicious: [], critical: [] };
        const summary = generatePreReviewSummary(result as never);
        expect(summary).toContain("1 条正常");
    });

    it("handles empty records", () => {
        const result = preReview([], sampleContext);
        expect(result.normal).toHaveLength(0);
        expect(result.suspicious).toHaveLength(0);
        expect(result.critical).toHaveLength(0);
        const summary = generatePreReviewSummary(result);
        expect(summary).toBe("暂无待审核记录");
    });
});
