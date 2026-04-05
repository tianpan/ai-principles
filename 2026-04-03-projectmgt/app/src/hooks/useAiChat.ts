"use client";

import { useState, useCallback } from "react";
import { MockLLMClient } from "../lib/ai/mock-llm-client";
import { buildChatContext } from "../lib/ai/context-manager";
import type { ChatMessage, ParsedIntent } from "../lib/ai/types";

interface UseAiChatReturn {
    sessionId: string;
    messages: ChatMessage[];
    loading: boolean;
    error: string | null;
    lastIntent: ParsedIntent | null;
    sendMessage: (content: string) => Promise<void>;
    confirmSubmit: () => void;
    resetSession: () => void;
}

const llmClient = new MockLLMClient();

function generateSessionId(): string {
    return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function useAiChat(
    projectContext: {
        projectId: number;
        projectName: string;
        workItems: Array<{
            name: string;
            targetQuantity: number;
            unit: string;
            completedQuantity: number;
        }>;
    } | null,
    onConfirm?: (intent: ParsedIntent) => void
): UseAiChatReturn {
    const [sessionId, setSessionId] = useState(generateSessionId);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [lastIntent, setLastIntent] = useState<ParsedIntent | null>(null);

    const sendMessage = useCallback(
        async (content: string) => {
            if (!projectContext) return;

            const userMessage: ChatMessage = { role: "user", content };
            setMessages((prev) => [...prev, userMessage]);
            setLoading(true);
            setError(null);

            try {
                const contextMessages = buildChatContext({
                    projectContext: {
                        projectId: projectContext.projectId,
                        projectName: projectContext.projectName,
                        workItems: projectContext.workItems,
                        date: new Date().toISOString().split("T")[0],
                    },
                    history: messages.filter((m) => m.role !== "system"),
                    userMessage: content,
                });

                const response = await llmClient.chat(contextMessages);

                const assistantMessage: ChatMessage = {
                    role: "assistant",
                    content: response.content,
                };
                setMessages((prev) => [...prev, assistantMessage]);

                if (response.parsedIntent) {
                    setLastIntent(response.parsedIntent);
                }
            } catch (err) {
                const msg = err instanceof Error ? err.message : "AI 助手暂时不可用";
                setError(msg);
                setMessages((prev) => [
                    ...prev,
                    { role: "assistant", content: `抱歉，出了点问题：${msg}。您可以手动填写表单。` },
                ]);
            } finally {
                setLoading(false);
            }
        },
        [projectContext, messages]
    );

    const confirmSubmit = useCallback(() => {
        if (lastIntent && onConfirm) {
            onConfirm(lastIntent);
            setMessages((prev) => [
                ...prev,
                { role: "assistant", content: "已提交成功！记录已创建。" },
            ]);
        }
        setLastIntent(null);
    }, [lastIntent, onConfirm]);

    const resetSession = useCallback(() => {
        setSessionId(generateSessionId());
        setMessages([]);
        setError(null);
        setLastIntent(null);
    }, []);

    return {
        sessionId,
        messages,
        loading,
        error,
        lastIntent,
        sendMessage,
        confirmSubmit,
        resetSession,
    };
}
