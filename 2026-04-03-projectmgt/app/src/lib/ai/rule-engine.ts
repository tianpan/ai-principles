/**
 * AI 智能审核规则引擎
 * 6 条异常检测规则，每条是纯函数：(record, context) => AuditResult | null
 */

/** 审核结果 */
export interface AuditResult {
    ruleId: string;
    ruleName: string;
    severity: "正常" | "疑似异常" | "明显问题";
    reason: string;
    suggestion: string;
}

/** 审核上下文 */
export interface AuditContext {
    history: Array<{
        date: string;
        moduleType: string;
        progress?: Array<Record<string, unknown>>;
        attendance?: Record<string, unknown>;
        materials?: Array<Record<string, unknown>>;
    }>;
    project: {
        id: number;
        name: string;
        startDate: string;
        endDate: string;
        workItems: Array<{
            name: string;
            targetQuantity: number;
            unit: string;
            weight: number;
        }>;
    };
    todayAttendance: {
        attendeeCount: number;
    } | null;
    workers: Array<{
        name: string;
        position: string;
        certifications: Array<{
            type: string;
            level: string;
            expireDate: string;
        }>;
    }>;
}

/** 每日记录简化类型 */
export interface AuditRecord {
    id: number;
    projectId: number;
    date: string;
    moduleType: string;
    status: string;
    progress?: Array<Record<string, unknown>>;
    attendance?: Record<string, unknown>;
    materials?: Array<Record<string, unknown>>;
    siteRecord?: Record<string, unknown>;
}

/** 可配置阈值常量 */
export const THRESHOLDS = {
    /** spike: 数据突变倍数 */
    SPIKE_MULTIPLIER: 3,
    /** attendance_mismatch: 考勤矛盾倍数 */
    ATTENDANCE_MISMATCH_MULTIPLIER: 1.5,
    /** 人均工效（假设值） */
    PRODUCTIVITY_PER_PERSON: 10,
    /** material_anomaly: 材料异常倍数 */
    MATERIAL_ANOMALY_MULTIPLIER: 1.2,
    /** progress_deviation: 进度偏离百分比 */
    PROGRESS_DEVIATION_PERCENT: 20,
};

/** 审核规则接口 */
export interface AuditRule {
    id: string;
    name: string;
    check(record: AuditRecord, context: AuditContext): AuditResult | null;
}

// ── Rule 1: spike（数据突变） ──────────────────────────────

export const spikeRule: AuditRule = {
    id: "spike",
    name: "数据突变",
    check(record, context): AuditResult | null {
        if (record.moduleType !== "进度" || !record.progress) return null;

        const todayDate = record.date;
        const recentRecords = context.history.filter(
            (h) => h.moduleType === "进度" && h.date !== todayDate
        );
        if (recentRecords.length < 3) return null;

        // 计算日均
        let totalQuantity = 0;
        let count = 0;
        for (const h of recentRecords) {
            if (h.progress) {
                for (const p of h.progress) {
                    if (typeof p.completedQuantity === "number") {
                        totalQuantity += p.completedQuantity;
                        count++;
                    }
                }
            }
        }
        if (count === 0) return null;
        const dailyAvg = totalQuantity / count;

        // 检查当前记录
        let currentTotal = 0;
        for (const p of record.progress) {
            if (typeof p.completedQuantity === "number") {
                currentTotal += p.completedQuantity;
            }
        }

        if (dailyAvg > 0 && currentTotal > dailyAvg * THRESHOLDS.SPIKE_MULTIPLIER) {
            return {
                ruleId: "spike",
                ruleName: "数据突变",
                severity: "疑似异常",
                reason: `当日完成量 ${currentTotal.toFixed(1)} 是近${recentRecords.length}天日均 ${dailyAvg.toFixed(1)} 的 ${(currentTotal / dailyAvg).toFixed(1)} 倍`,
                suggestion: "请核实当日工程量数据是否正确",
            };
        }
        return null;
    },
};

// ── Rule 2: attendance_mismatch（考勤矛盾） ───────────────

export const attendanceMismatchRule: AuditRule = {
    id: "attendance_mismatch",
    name: "考勤矛盾",
    check(record, context): AuditResult | null {
        if (record.moduleType !== "进度" || !record.progress) return null;
        if (!context.todayAttendance) return null;

        const attendeeCount = context.todayAttendance.attendeeCount;
        if (attendeeCount === 0) return null;

        let currentTotal = 0;
        for (const p of record.progress) {
            if (typeof p.completedQuantity === "number") {
                currentTotal += p.completedQuantity;
            }
        }

        const maxExpected =
            attendeeCount * THRESHOLDS.PRODUCTIVITY_PER_PERSON * THRESHOLDS.ATTENDANCE_MISMATCH_MULTIPLIER;

        if (currentTotal > maxExpected) {
            return {
                ruleId: "attendance_mismatch",
                ruleName: "考勤矛盾",
                severity: "明显问题",
                reason: `出勤 ${attendeeCount} 人，合理完成量上限 ${maxExpected}，实际完成 ${currentTotal.toFixed(1)}`,
                suggestion: "请核实考勤人数或工程量数据",
            };
        }
        return null;
    },
};

// ── Rule 3: material_anomaly（材料异常） ──────────────────

export const materialAnomalyRule: AuditRule = {
    id: "material_anomaly",
    name: "材料异常",
    check(record): AuditResult | null {
        if (record.moduleType !== "材料" || !record.materials) return null;

        for (const mat of record.materials) {
            if (typeof mat.quantity !== "number") continue;
            const remainingDemand = mat.remainingDemand as number | undefined;
            if (remainingDemand === undefined || remainingDemand <= 0) continue;

            if (mat.quantity > remainingDemand * THRESHOLDS.MATERIAL_ANOMALY_MULTIPLIER) {
                return {
                    ruleId: "material_anomaly",
                    ruleName: "材料异常",
                    severity: "疑似异常",
                    reason: `${mat.materialName ?? "材料"} 领料量 ${mat.quantity} 超过剩余需求 ${remainingDemand} 的 ${THRESHOLDS.MATERIAL_ANOMALY_MULTIPLIER} 倍`,
                    suggestion: "请核实领料数量是否正确",
                };
            }
        }
        return null;
    },
};

// ── Rule 4: compliance_gap（合规缺失） ────────────────────

/** 高风险作业关键词 */
const HIGH_RISK_KEYWORDS = ["高空", "动火", "有限空间", "深基坑", "吊装"];

export const complianceGapRule: AuditRule = {
    id: "compliance_gap",
    name: "合规缺失",
    check(record): AuditResult | null {
        if (record.moduleType !== "施工记录" && record.moduleType !== "考勤") return null;

        const siteRecord = record.siteRecord;
        if (!siteRecord) return null;

        const description = (siteRecord.description as string) ?? "";
        const hasHighRisk = HIGH_RISK_KEYWORDS.some((kw) => description.includes(kw));
        if (!hasHighRisk) return null;

        const hasSafetyBriefing = siteRecord.safetyBriefingType !== undefined
            && siteRecord.safetyBriefingType !== null;

        if (!hasSafetyBriefing) {
            return {
                ruleId: "compliance_gap",
                ruleName: "合规缺失",
                severity: "明显问题",
                reason: "涉及高风险作业但无安全交底记录",
                suggestion: "请补充安全交底记录",
            };
        }
        return null;
    },
};

// ── Rule 5: cert_expired（证件过期） ──────────────────────

export const certExpiredRule: AuditRule = {
    id: "cert_expired",
    name: "证件过期",
    check(record, context): AuditResult | null {
        if (record.moduleType !== "考勤" || !record.attendance) return null;

        const attendeeIds = record.attendance.attendeeIds as number[] | undefined;
        if (!attendeeIds || attendeeIds.length === 0) return null;

        const today = new Date().toISOString().split("T")[0];
        const expiredWorkers: string[] = [];

        for (const worker of context.workers) {
            for (const cert of worker.certifications) {
                if (cert.expireDate < today) {
                    expiredWorkers.push(`${worker.name}(${cert.type}, 已过期 ${cert.expireDate})`);
                }
            }
        }

        if (expiredWorkers.length > 0) {
            return {
                ruleId: "cert_expired",
                ruleName: "证件过期",
                severity: "明显问题",
                reason: `证件过期人员：${expiredWorkers.join("、")}`,
                suggestion: "请安排相关人员进行证件续期",
            };
        }
        return null;
    },
};

// ── Rule 6: progress_deviation（进度偏离） ────────────────

export const progressDeviationRule: AuditRule = {
    id: "progress_deviation",
    name: "进度偏离",
    check(record, context): AuditResult | null {
        if (record.moduleType !== "进度" || !record.progress) return null;

        const { startDate, endDate } = context.project;
        const recordDate = new Date(record.date);
        const start = new Date(startDate);
        const end = new Date(endDate);
        const totalDays = (end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24);
        const elapsedDays = (recordDate.getTime() - start.getTime()) / (1000 * 60 * 60 * 24);

        if (totalDays <= 0 || elapsedDays < 0) return null;

        const plannedPercent = Math.min((elapsedDays / totalDays) * 100, 100);

        // 计算实际进度（加权）
        let actualPercent = 0;
        for (const wi of context.project.workItems) {
            const cumulative = record.progress.find(
                (p) => p.workItemName === wi.name
            );
            const completed = typeof cumulative?.completedQuantity === "number"
                ? cumulative.completedQuantity : 0;
            const itemPercent = wi.targetQuantity > 0
                ? (completed / wi.targetQuantity) * 100 : 0;
            actualPercent += (itemPercent * wi.weight) / 100;
        }

        const deviation = Math.abs(actualPercent - plannedPercent);
        if (deviation > THRESHOLDS.PROGRESS_DEVIATION_PERCENT) {
            const direction = actualPercent < plannedPercent ? "滞后" : "超前";
            return {
                ruleId: "progress_deviation",
                ruleName: "进度偏离",
                severity: "疑似异常",
                reason: `实际进度 ${actualPercent.toFixed(1)}%，计划进度 ${plannedPercent.toFixed(1)}%，${direction} ${deviation.toFixed(1)}%`,
                suggestion: direction === "滞后"
                    ? "建议增加人手或调整施工计划"
                    : "请确认数据准确性",
            };
        }
        return null;
    },
};

// ── 规则注册表 ────────────────────────────────────────────

export const ALL_RULES: AuditRule[] = [
    spikeRule,
    attendanceMismatchRule,
    materialAnomalyRule,
    complianceGapRule,
    certExpiredRule,
    progressDeviationRule,
];

/** 运行所有规则，返回命中的结果列表 */
export function runAudit(
    record: AuditRecord,
    context: AuditContext
): AuditResult[] {
    return ALL_RULES
        .map((rule) => rule.check(record, context))
        .filter((r): r is AuditResult => r !== null);
}
