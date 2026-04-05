import type { ProjectContext } from "./types";

/** 对话助手 System Prompt */
export function getChatSystemPrompt(context: ProjectContext): string {
    const workItemList = context.workItems
        .map((w) => `  - ${w.name}: 目标 ${w.targetQuantity}${w.unit}, 已完成 ${w.completedQuantity}${w.unit}`)
        .join("\n");

    return `你是"工程管家"AI助手，帮助班组长快速登记每日施工数据。

当前项目：${context.projectName}
日期：${context.date}

工作项：
${workItemList}

你可以识别以下模块的输入：
1. **进度** — 如"今天焊了80米"、"开挖50米"
2. **材料** — 如"领了30根管"、"退料20米"
3. **考勤** — 如"签到3人"、"来了5个人"
4. **施工记录** — 如"记录施工，K3+500段"

请根据用户输入识别意图，提取关键数据，并生成确认回复。`;
}

/** 每日简报 Prompt */
export function getBriefingPrompt(data: {
    projectName: string;
    date: string;
    progressSummary: string;
    materialSummary: string;
    attendanceSummary: string;
    risks: string[];
}): string {
    const riskSection = data.risks.length > 0
        ? `\n风险提示：\n${data.risks.map((r) => `- ${r}`).join("\n")}`
        : "\n风险提示：无";

    return `# 每日施工简报

**项目**：${data.projectName}
**日期**：${data.date}

## 进度概况
${data.progressSummary}

## 材料动态
${data.materialSummary}

## 人员出勤
${data.attendanceSummary}
${riskSection}

---
*本简报由 AI 自动生成，仅供参考。`;
}

/** 自然语言查询 Prompt（Mock 模式下用于识别查询类型） */
export function getNLQueryPrompt(question: string): string {
    return `用户提问："${question}"

请识别这是一个关于什么类型的查询：
1. 进度查询（如"还剩多少没焊"）
2. 材料查询（如"领了多少管"）
3. 考勤查询（如"今天来了几个人"）
4. 综合查询（其他）`;
}
