import { describe, it, expect } from "vitest";
import {
    spikeRule,
    attendanceMismatchRule,
    materialAnomalyRule,
    complianceGapRule,
    certExpiredRule,
    progressDeviationRule,
    runAudit,
    ALL_RULES,
    type AuditRecord,
    type AuditContext,
} from "./rule-engine";

function makeContext(overrides: Partial<AuditContext> = {}): AuditContext {
    return {
        history: [],
        project: {
            id: 1,
            name: "XX路中压燃气管道工程",
            startDate: "2026-03-01",
            endDate: "2026-05-31",
            workItems: [
                { name: "中压管线焊接", targetQuantity: 1200, unit: "米", weight: 35 },
                { name: "管沟开挖", targetQuantity: 1500, unit: "米", weight: 20 },
            ],
        },
        todayAttendance: { attendeeCount: 5 },
        workers: [],
        ...overrides,
    };
}

function makeRecord(overrides: Partial<AuditRecord> = {}): AuditRecord {
    return {
        id: 1,
        projectId: 1,
        date: "2026-04-03",
        moduleType: "进度",
        status: "待审核",
        ...overrides,
    };
}

// ── spike ──────────────────────────────────────────────────

describe("spikeRule", () => {
    it("returns null for non-progress records", () => {
        const result = spikeRule.check(
            makeRecord({ moduleType: "材料" }),
            makeContext()
        );
        expect(result).toBeNull();
    });

    it("returns null when history is too short", () => {
        const result = spikeRule.check(
            makeRecord({ progress: [{ completedQuantity: 500 }] }),
            makeContext({ history: [{ date: "2026-04-02", moduleType: "进度", progress: [{ completedQuantity: 80 }] }] })
        );
        expect(result).toBeNull();
    });

    it("detects spike when quantity > 3x average", () => {
        const history = Array.from({ length: 7 }, (_, i) => ({
            date: `2026-03-${27 + i}`,
            moduleType: "进度",
            progress: [{ completedQuantity: 80 }],
        }));
        const result = spikeRule.check(
            makeRecord({ progress: [{ completedQuantity: 300 }] }),
            makeContext({ history })
        );
        expect(result).not.toBeNull();
        expect(result!.ruleId).toBe("spike");
        expect(result!.severity).toBe("疑似异常");
    });

    it("returns null when quantity is within normal range", () => {
        const history = Array.from({ length: 7 }, (_, i) => ({
            date: `2026-03-${27 + i}`,
            moduleType: "进度",
            progress: [{ completedQuantity: 80 }],
        }));
        const result = spikeRule.check(
            makeRecord({ progress: [{ completedQuantity: 100 }] }),
            makeContext({ history })
        );
        expect(result).toBeNull();
    });
});

// ── attendance_mismatch ────────────────────────────────────

describe("attendanceMismatchRule", () => {
    it("detects mismatch when quantity exceeds expected", () => {
        const result = attendanceMismatchRule.check(
            makeRecord({ progress: [{ completedQuantity: 200 }] }),
            makeContext({ todayAttendance: { attendeeCount: 2 } })
        );
        expect(result).not.toBeNull();
        expect(result!.ruleId).toBe("attendance_mismatch");
        expect(result!.severity).toBe("明显问题");
    });

    it("returns null when quantity is reasonable", () => {
        const result = attendanceMismatchRule.check(
            makeRecord({ progress: [{ completedQuantity: 40 }] }),
            makeContext({ todayAttendance: { attendeeCount: 5 } })
        );
        expect(result).toBeNull();
    });

    it("returns null when no attendance data", () => {
        const result = attendanceMismatchRule.check(
            makeRecord({ progress: [{ completedQuantity: 200 }] }),
            makeContext({ todayAttendance: null })
        );
        expect(result).toBeNull();
    });
});

// ── material_anomaly ───────────────────────────────────────

describe("materialAnomalyRule", () => {
    it("detects material anomaly", () => {
        const result = materialAnomalyRule.check(
            makeRecord({
                moduleType: "材料",
                materials: [{
                    materialName: "DN100镀锌钢管",
                    quantity: 100,
                    remainingDemand: 50,
                }],
            }),
            makeContext()
        );
        expect(result).not.toBeNull();
        expect(result!.ruleId).toBe("material_anomaly");
    });

    it("returns null when quantity is within demand", () => {
        const result = materialAnomalyRule.check(
            makeRecord({
                moduleType: "材料",
                materials: [{
                    materialName: "DN100镀锌钢管",
                    quantity: 30,
                    remainingDemand: 200,
                }],
            }),
            makeContext()
        );
        expect(result).toBeNull();
    });
});

// ── compliance_gap ─────────────────────────────────────────

describe("complianceGapRule", () => {
    it("detects missing safety briefing for high-risk work", () => {
        const result = complianceGapRule.check(
            makeRecord({
                moduleType: "施工记录",
                siteRecord: {
                    description: "高空作业区域管线更换",
                },
            }),
            makeContext()
        );
        expect(result).not.toBeNull();
        expect(result!.ruleId).toBe("compliance_gap");
        expect(result!.severity).toBe("明显问题");
    });

    it("returns null when safety briefing is present", () => {
        const result = complianceGapRule.check(
            makeRecord({
                moduleType: "施工记录",
                siteRecord: {
                    description: "高空作业区域管线更换",
                    safetyBriefingType: "高空作业安全交底",
                },
            }),
            makeContext()
        );
        expect(result).toBeNull();
    });

    it("returns null for non-high-risk work", () => {
        const result = complianceGapRule.check(
            makeRecord({
                moduleType: "施工记录",
                siteRecord: { description: "普通管段焊接" },
            }),
            makeContext()
        );
        expect(result).toBeNull();
    });
});

// ── cert_expired ───────────────────────────────────────────

describe("certExpiredRule", () => {
    it("detects expired certifications", () => {
        const result = certExpiredRule.check(
            makeRecord({
                moduleType: "考勤",
                attendance: { attendeeIds: [1, 2] },
            }),
            makeContext({
                workers: [{
                    name: "李四",
                    position: "焊工",
                    certifications: [{
                        type: "焊接作业证",
                        level: "初级",
                        expireDate: "2026-01-01",
                    }],
                }],
            })
        );
        expect(result).not.toBeNull();
        expect(result!.ruleId).toBe("cert_expired");
    });

    it("returns null when all certs are valid", () => {
        const result = certExpiredRule.check(
            makeRecord({
                moduleType: "考勤",
                attendance: { attendeeIds: [1] },
            }),
            makeContext({
                workers: [{
                    name: "李四",
                    position: "焊工",
                    certifications: [{
                        type: "焊接作业证",
                        level: "初级",
                        expireDate: "2027-12-31",
                    }],
                }],
            })
        );
        expect(result).toBeNull();
    });
});

// ── progress_deviation ─────────────────────────────────────

describe("progressDeviationRule", () => {
    it("detects progress lag exceeding 20%", () => {
        // 项目 3/1-5/31 (92天), 4/3 = 第33天 → plannedPercent ≈ 36%
        // actualPercent = 5% → deviation ≈ 31% > 20%
        const result = progressDeviationRule.check(
            makeRecord({
                progress: [{ workItemName: "中压管线焊接", completedQuantity: 60 }],
            }),
            makeContext()
        );
        expect(result).not.toBeNull();
        expect(result!.ruleId).toBe("progress_deviation");
    });

    it("returns null when progress is on track", () => {
        // actualPercent 接近 plannedPercent
        const result = progressDeviationRule.check(
            makeRecord({
                progress: [
                    { workItemName: "中压管线焊接", completedQuantity: 500 },
                    { workItemName: "管沟开挖", completedQuantity: 600 },
                ],
            }),
            makeContext()
        );
        expect(result).toBeNull();
    });
});

// ── runAudit ───────────────────────────────────────────────

describe("runAudit", () => {
    it("runs all 6 rules", () => {
        expect(ALL_RULES).toHaveLength(6);
    });

    it("returns empty array when no rules trigger", () => {
        const results = runAudit(
            makeRecord({
                moduleType: "材料",
                materials: [{ quantity: 10, remainingDemand: 200 }],
            }),
            makeContext()
        );
        expect(results).toEqual([]);
    });

    it("returns multiple results when multiple rules trigger", () => {
        const history = Array.from({ length: 7 }, (_, i) => ({
            date: `2026-03-${27 + i}`,
            moduleType: "进度",
            progress: [{ completedQuantity: 80 }],
        }));
        const results = runAudit(
            makeRecord({ progress: [{ completedQuantity: 300 }] }),
            makeContext({
                history,
                todayAttendance: { attendeeCount: 1 },
            })
        );
        // Should trigger both spike and attendance_mismatch
        expect(results.length).toBeGreaterThanOrEqual(2);
        const ruleIds = results.map((r) => r.ruleId);
        expect(ruleIds).toContain("spike");
        expect(ruleIds).toContain("attendance_mismatch");
    });
});
