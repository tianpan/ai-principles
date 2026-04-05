import type { ChatMessage, ProjectContext } from "./types";
import { getChatSystemPrompt } from "./prompts";

const CONTEXT_WINDOW_SIZE = 10;

/** 组装对话上下文：system prompt + 项目数据 + 历史对话 */
export function buildChatContext(params: {
    projectContext: ProjectContext;
    history: ChatMessage[];
    userMessage: string;
}): ChatMessage[] {
    const { projectContext, history, userMessage } = params;

    // system prompt
    const systemMessage: ChatMessage = {
        role: "system",
        content: getChatSystemPrompt(projectContext),
    };

    // 截取最近 N 条历史（不含 system）
    const recentHistory = history.slice(-CONTEXT_WINDOW_SIZE);

    // 当前用户消息
    const currentMessage: ChatMessage = {
        role: "user",
        content: userMessage,
    };

    return [systemMessage, ...recentHistory, currentMessage];
}

/** 从 DailyRecord 提取项目上下文 */
export function extractProjectContext(params: {
    projectName: string;
    projectId: number;
    workItems: Array<{
        name: string;
        targetQuantity: number;
        unit: string;
    }>;
    dailyRecords: Array<{
        progress?: Array<Record<string, unknown>>;
    }>;
    date: string;
}): ProjectContext {
    const { projectName, projectId, workItems, dailyRecords, date } = params;

    // 计算每个工作项累计完成量
    const workItemsWithProgress = workItems.map((wi) => {
        let completedQuantity = 0;
        for (const record of dailyRecords) {
            if (record.progress) {
                for (const p of record.progress) {
                    if (p.workItemName === wi.name && typeof p.completedQuantity === "number") {
                        completedQuantity += p.completedQuantity;
                    }
                }
            }
        }
        return {
            name: wi.name,
            targetQuantity: wi.targetQuantity,
            unit: wi.unit,
            completedQuantity,
        };
    });

    return {
        projectId,
        projectName,
        workItems: workItemsWithProgress,
        date,
    };
}
