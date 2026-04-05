/** NL 查询类型 */
export type QueryType = "进度" | "材料" | "考勤" | "综合";

/** NL 查询结果 */
export interface NLQueryResult {
    question: string;
    queryType: QueryType;
    answer: string;
    data: Array<{ label: string; value: string }>;
}

/** 查询类型关键词 */
const QUERY_KEYWORDS: Record<QueryType, string[]> = {
    进度: ["进度", "完成", "还剩", "焊", "铺", "挖", "装", "米", "根", "户", "百分", "比例"],
    材料: ["材料", "领", "退", "料", "管", "设备", "库存", "用量"],
    考勤: ["考勤", "签到", "人数", "出勤", "人员", "工人", "来了几"],
    综合: [],
};

/** 识别查询类型 */
function identifyQueryType(question: string): QueryType {
    const scores: Record<string, number> = {};
    for (const [type, keywords] of Object.entries(QUERY_KEYWORDS)) {
        scores[type] = keywords.reduce(
            (sum, kw) => sum + (question.includes(kw) ? 1 : 0),
            0
        );
    }
    const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
    return best[1] > 0 ? (best[0] as QueryType) : "综合";
}

/** 数据接口 — 由调用方从 MockDataProvider 获取 */
export interface QueryDataContext {
    projectName: string;
    workItems: Array<{
        name: string;
        targetQuantity: number;
        completedQuantity: number;
        unit: string;
    }>;
    todayAttendance: number;
    materialSummary: Array<{
        name: string;
        type: string;
        quantity: number;
        unit: string;
    }>;
    totalRecords: number;
    approvedRecords: number;
}

/** 生成自然语言回答 */
function generateAnswer(question: string, queryType: QueryType, ctx: QueryDataContext): {
    answer: string;
    data: Array<{ label: string; value: string }>;
} {
    const data: Array<{ label: string; value: string }> = [];

    switch (queryType) {
        case "进度": {
            const lines = ctx.workItems.map((wi) => {
                const pct = wi.targetQuantity > 0
                    ? Math.round((wi.completedQuantity / wi.targetQuantity) * 100)
                    : 0;
                data.push({
                    label: wi.name,
                    value: `${wi.completedQuantity}/${wi.targetQuantity}${wi.unit} (${pct}%)`,
                });
                return `${wi.name}：已完成 ${wi.completedQuantity}${wi.unit}，目标 ${wi.targetQuantity}${wi.unit}，完成 ${pct}%`;
            });
            return {
                answer: `${ctx.projectName}当前进度：\n${lines.join("\n")}`,
                data,
            };
        }
        case "材料": {
            if (ctx.materialSummary.length === 0) {
                return { answer: "今日暂无材料记录。", data: [] };
            }
            const lines = ctx.materialSummary.map((m) => {
                data.push({ label: m.name, value: `${m.type} ${m.quantity}${m.unit}` });
                return `${m.type}：${m.name} ${m.quantity}${m.unit}`;
            });
            return {
                answer: `今日材料动态：\n${lines.join("\n")}`,
                data,
            };
        }
        case "考勤": {
            data.push({ label: "今日出勤", value: `${ctx.todayAttendance}人` });
            return {
                answer: `今日出勤 ${ctx.todayAttendance} 人。`,
                data,
            };
        }
        default: {
            data.push({ label: "项目", value: ctx.projectName });
            data.push({ label: "总记录", value: `${ctx.totalRecords}条` });
            data.push({ label: "已审核", value: `${ctx.approvedRecords}条` });
            return {
                answer: `${ctx.projectName} 综合概况：共 ${ctx.totalRecords} 条记录，已审核 ${ctx.approvedRecords} 条。`,
                data,
            };
        }
    }
}

/** 执行 NL 查询 */
export function executeNLQuery(
    question: string,
    context: QueryDataContext
): NLQueryResult {
    const queryType = identifyQueryType(question);
    const { answer, data } = generateAnswer(question, queryType, context);
    return { question, queryType, answer, data };
}
