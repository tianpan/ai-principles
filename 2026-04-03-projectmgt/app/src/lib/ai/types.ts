/**
 * AI 服务层类型定义
 */

/** 对话消息 */
export interface ChatMessage {
    role: "user" | "assistant" | "system";
    content: string;
}

/** LLM 请求选项 */
export interface LLMOptions {
    maxTokens?: number;
    temperature?: number;
}

/** LLM 响应 */
export interface LLMResponse {
    content: string;
    parsedIntent?: ParsedIntent;
    needConfirm?: boolean;
}

/** LLM 流式输出块 */
export interface LLMChunk {
    content: string;
    done: boolean;
}

/** 意图识别结果 */
export interface ParsedIntent {
    module: "考勤" | "进度" | "材料" | "施工记录";
    action: "create" | "query" | "update";
    fields: Record<string, unknown>;
}

/** LLM 客户端接口 */
export interface LLMClient {
    chat(messages: ChatMessage[], options?: LLMOptions): Promise<LLMResponse>;
}

/** 项目上下文（注入到 prompt） */
export interface ProjectContext {
    projectId: number;
    projectName: string;
    workItems: Array<{
        name: string;
        targetQuantity: number;
        unit: string;
        completedQuantity: number;
    }>;
    date: string;
}
