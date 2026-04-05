import { runAudit, type AuditRecord, type AuditContext, type AuditResult } from "./rule-engine";

/** 预审分类结果 */
export interface PreReviewResult {
    normal: AuditRecord[];
    suspicious: Array<{ record: AuditRecord; results: AuditResult[] }>;
    critical: Array<{ record: AuditRecord; results: AuditResult[] }>;
}

/** 对待审核记录执行预审 */
export function preReview(
    records: AuditRecord[],
    context: AuditContext
): PreReviewResult {
    const normal: AuditRecord[] = [];
    const suspicious: Array<{ record: AuditRecord; results: AuditResult[] }> = [];
    const critical: Array<{ record: AuditRecord; results: AuditResult[] }> = [];

    for (const record of records) {
        const results = runAudit(record, context);

        if (results.length === 0) {
            normal.push(record);
        } else {
            const maxSeverity = results.reduce((max, r) => {
                if (r.severity === "明显问题") return "明显问题";
                if (r.severity === "疑似异常" && max !== "明显问题") return "疑似异常";
                return max;
            }, "正常" as string);

            if (maxSeverity === "明显问题") {
                critical.push({ record, results });
            } else {
                suspicious.push({ record, results });
            }
        }
    }

    return { normal, suspicious, critical };
}

/** 生成预审摘要文本 */
export function generatePreReviewSummary(result: PreReviewResult): string {
    const parts: string[] = [];
    if (result.normal.length > 0) {
        parts.push(`✅ ${result.normal.length} 条正常`);
    }
    if (result.suspicious.length > 0) {
        parts.push(`🟡 ${result.suspicious.length} 条疑似异常`);
    }
    if (result.critical.length > 0) {
        parts.push(`🔴 ${result.critical.length} 条明显问题`);
    }
    return parts.join(" | ") || "暂无待审核记录";
}
