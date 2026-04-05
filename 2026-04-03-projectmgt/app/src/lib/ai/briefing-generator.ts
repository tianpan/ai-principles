import { getBriefingPrompt } from "./prompts";

/** 简报数据输入 */
export interface BriefingInput {
    projectId: number;
    projectName: string;
    date: string;
    progressItems: Array<{
        name: string;
        todayQuantity: number;
        cumulativeQuantity: number;
        targetQuantity: number;
        unit: string;
    }>;
    materialSummary: string;
    attendanceSummary: string;
    risks: string[];
}

/** 简报生成输出 */
export interface BriefingOutput {
    projectId: number;
    date: string;
    content: string;
    risks: Array<{
        type: string;
        severity: "高" | "中" | "低";
        description: string;
    }>;
}

/** 根据风险文本推断严重等级 */
function inferSeverity(riskText: string): "高" | "中" | "低" {
    if (/严重|超期|重大|紧急/.test(riskText)) return "高";
    if (/滞后|不足|偏差|注意/.test(riskText)) return "中";
    return "低";
}

/** 推断风险类型 */
function inferRiskType(riskText: string): string {
    if (/进度|滞后|工期/.test(riskText)) return "进度滞后";
    if (/材料|缺|不足/.test(riskText)) return "材料短缺";
    if (/安全|风险|隐患/.test(riskText)) return "安全风险";
    if (/人员|人手/.test(riskText)) return "人员不足";
    return "其他";
}

/** 生成每日施工简报 */
export function generateBriefing(input: BriefingInput): BriefingOutput {
    const progressSummary = input.progressItems
        .map((item) => {
            const pct = item.targetQuantity > 0
                ? Math.round((item.cumulativeQuantity / item.targetQuantity) * 100)
                : 0;
            return `- ${item.name}：今日完成 ${item.todayQuantity}${item.unit}，累计 ${item.cumulativeQuantity}/${item.targetQuantity}${item.unit} (${pct}%)`;
        })
        .join("\n");

    const content = getBriefingPrompt({
        projectName: input.projectName,
        date: input.date,
        progressSummary: progressSummary || "暂无进度数据",
        materialSummary: input.materialSummary || "暂无材料数据",
        attendanceSummary: input.attendanceSummary || "暂无考勤数据",
        risks: input.risks,
    });

    const risks = input.risks.map((r) => ({
        type: inferRiskType(r),
        severity: inferSeverity(r),
        description: r,
    }));

    return {
        projectId: input.projectId,
        date: input.date,
        content,
        risks,
    };
}
