import type { ChatMessage, LLMClient, LLMOptions, LLMResponse, ParsedIntent } from "./types";

/** 模块关键词表 */
const MODULE_KEYWORDS: Record<ParsedIntent["module"], string[]> = {
    进度: ["焊", "铺", "挖", "装", "米", "根", "户", "完成"],
    材料: ["领", "退", "料", "管", "设备"],
    考勤: ["签", "到", "来", "没来", "人"],
    施工记录: ["记录", "施工", "安装", "开挖"],
};

/** 数字提取正则 */
const NUMBER_REGEX = /(\d+(?:\.\d+)?)/g;

/** 单位映射 */
const UNIT_MAP: Record<string, string> = {
    米: "米", m: "米",
    根: "根", 个: "个", 户: "户",
    台班: "台班", 吨: "吨",
};

/** 从用户输入匹配模块 */
function matchModule(input: string): ParsedIntent["module"] | null {
    const scores: Record<string, number> = {};
    for (const [module, keywords] of Object.entries(MODULE_KEYWORDS)) {
        scores[module] = keywords.reduce(
            (sum, kw) => sum + (input.includes(kw) ? 1 : 0),
            0
        );
    }
    const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
    return best[1] > 0 ? (best[0] as ParsedIntent["module"]) : null;
}

/** 从用户输入提取数字 */
function extractNumbers(input: string): number[] {
    const matches = input.match(NUMBER_REGEX);
    return matches ? matches.map(Number) : [];
}

/** 从用户输入提取单位 */
function extractUnit(input: string): string {
    for (const [key, unit] of Object.entries(UNIT_MAP)) {
        if (input.includes(key)) return unit;
    }
    return "";
}

/** 解析用户意图 */
function parseIntent(input: string): ParsedIntent | null {
    const matchedModule = matchModule(input);
    if (!matchedModule) return null;

    const numbers = extractNumbers(input);
    const unit = extractUnit(input);
    const fields: Record<string, unknown> = {};

    if (numbers.length > 0) {
        fields.quantity = numbers[0];
    }
    if (unit) {
        fields.unit = unit;
    }

    // 模块特定字段提取
    if (matchedModule === "进度" && numbers.length > 0) {
        fields.completedQuantity = numbers[0];
    } else if (matchedModule === "材料") {
        if (input.includes("领")) fields.recordType = "领料";
        else if (input.includes("退")) fields.recordType = "退料";
    } else if (matchedModule === "考勤") {
        if (numbers.length > 0) fields.attendeeCount = numbers[0];
    }

    return {
        module: matchedModule,
        action: "create",
        fields,
    };
}

/** 生成回复文本 */
function generateReply(intent: ParsedIntent | null): string {
    if (!intent) {
        return "我还没太理解您的意思。您可以试试这样说：\n- 「今天焊了80米」\n- 「领了30根管」\n- 「签到3人」\n- 「记录施工，K3+500段」";
    }

    const { module, fields } = intent;
    const quantity = fields.quantity ?? fields.completedQuantity;
    const unit = fields.unit ?? "";

    if (module === "进度") {
        return `好的，已识别为**进度记录**：完成 ${quantity ?? ""}${unit}。确认后将创建今日进度记录，是否确认？`;
    }
    if (module === "材料") {
        const type = fields.recordType ?? "领料";
        return `好的，已识别为**材料${type}**：${quantity ?? ""}${unit}。确认后将创建材料记录，是否确认？`;
    }
    if (module === "考勤") {
        return `好的，已识别为**考勤签到**：${fields.attendeeCount ?? ""}人。确认后将创建考勤记录，是否确认？`;
    }
    if (module === "施工记录") {
        return `好的，已识别为**施工记录**。请补充桩号/位置信息，确认后创建记录。`;
    }

    return "请确认您的输入，我将帮您创建对应记录。";
}

/** Mock LLM 客户端 — 关键词匹配 + 模板生成 */
export class MockLLMClient implements LLMClient {
    async chat(
        messages: ChatMessage[],
        options?: LLMOptions
    ): Promise<LLMResponse> {
        void options;
        // 取最后一条用户消息
        const lastUserMsg = [...messages]
            .reverse()
            .find((m) => m.role === "user");
        const input = lastUserMsg?.content ?? "";

        const parsedIntent = parseIntent(input);
        const content = generateReply(parsedIntent);

        return {
            content,
            parsedIntent: parsedIntent ?? undefined,
            needConfirm: parsedIntent !== null,
        };
    }
}
