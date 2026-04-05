"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { AiChatBubble } from "./AiChatBubble";
import { AiChatQuickActions } from "./AiChatQuickActions";
import { useAiChat } from "../../hooks/useAiChat";
import { useMockData } from "../../lib/MockDataProvider";
import type { ParsedIntent } from "../../lib/ai/types";

const QUICK_ACTIONS = [
    { module: "安全考勤", icon: "👷" },
    { module: "施工进度", icon: "📊" },
    { module: "材料登记", icon: "📦" },
    { module: "施工记录", icon: "📝" },
];

export function AiChatWidget() {
    const [open, setOpen] = useState(false);
    const [mounted, setMounted] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);
    const scrollRef = useRef<HTMLDivElement>(null);
    const { currentUser, data, getProjectsByTeamLeader, getCumulativeProgress, addDailyRecord } = useMockData();

    // 构建项目上下文
    const projectContext = (() => {
        if (!currentUser) return null;
        const projects = getProjectsByTeamLeader(currentUser.id);
        if (projects.length === 0) return null;
        const project = projects[0];
        const workItems = project.workItems.map((wi) => ({
            name: wi.name,
            targetQuantity: wi.targetQuantity,
            unit: wi.unit,
            completedQuantity: getCumulativeProgress(project.id, wi.id),
        }));
        return {
            projectId: project.id,
            projectName: project.name,
            workItems,
        };
    })();

    const handleConfirmIntent = useCallback((intent: ParsedIntent) => {
        if (!currentUser || !projectContext) return;
        const today = new Date().toISOString().split("T")[0];
        const baseRecord = {
            projectId: projectContext.projectId,
            teamLeaderId: currentUser.id,
            date: today,
            status: "待审核",
            reviewComment: null,
            reviewedAt: null,
        };

        const { module, fields } = intent;
        if (module === "进度") {
            const wi = projectContext.workItems[0];
            addDailyRecord({
                ...baseRecord,
                moduleType: "进度",
                progress: [{
                    workItemId: 1,
                    quantity: (fields.completedQuantity as number) ?? (fields.quantity as number) ?? 0,
                    workItemName: wi?.name ?? "",
                    unit: (fields.unit as string) ?? wi?.unit ?? "",
                }],
            });
        } else if (module === "材料") {
            addDailyRecord({
                ...baseRecord,
                moduleType: "材料",
                materials: [{
                    name: "材料",
                    recordType: (fields.recordType as string) ?? "领料",
                    quantity: (fields.quantity as number) ?? 0,
                    unit: (fields.unit as string) ?? "",
                }],
            });
        } else if (module === "考勤") {
            addDailyRecord({
                ...baseRecord,
                moduleType: "考勤",
                attendance: {
                    attendeeCount: (fields.attendeeCount as number) ?? 0,
                },
            });
        } else if (module === "施工记录") {
            addDailyRecord({
                ...baseRecord,
                moduleType: "施工记录",
                siteRecord: { description: "AI 助手创建" },
            });
        }
    }, [currentUser, projectContext, addDailyRecord]);

    const { messages, loading, lastIntent, sendMessage, confirmSubmit, resetSession } =
        useAiChat(projectContext, handleConfirmIntent);

    useEffect(() => {
        setMounted(true);
    }, []);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSend = () => {
        if (!inputRef.current?.value.trim()) return;
        sendMessage(inputRef.current.value.trim());
        inputRef.current.value = "";
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleQuickAction = (module: string) => {
        sendMessage(`帮我填写${module}`);
    };

    const handleFillAll = () => {
        sendMessage("帮我填写今天的所有表单");
    };

    // 跟踪已填模块（简化：根据今日 dailyRecords）
    const today = new Date().toISOString().split("T")[0];
    const filledModules = new Set(
        data.dailyRecords
            .filter((r) => r.date === today && r.status !== "已退回")
            .map((r) => r.moduleType)
    );

    if (!mounted) return null;

    const widget = (
        <>
            {/* 浮窗按钮 */}
            <button
                onClick={() => setOpen(!open)}
                className="fixed bottom-20 right-4 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-eng-blue text-white shadow-lg transition-transform active:scale-95"
                aria-label="AI 助手"
            >
                {open ? (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M18 6L6 18M6 6l12 12" />
                    </svg>
                ) : (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                    </svg>
                )}
            </button>

            {/* 对话面板 */}
            <div
                className={`fixed inset-x-0 bottom-0 z-50 transition-transform duration-300 ease-in-out ${
                    open ? "translate-y-0" : "translate-y-full"
                }`}
                style={{ maxHeight: "75vh" }}
            >
                <div className="mx-auto max-w-lg rounded-t-2xl bg-white shadow-2xl flex flex-col" style={{ maxHeight: "75vh" }}>
                    {/* 头部 */}
                    <div className="flex items-center justify-between border-b px-4 py-3">
                        <div>
                            <h3 className="text-base font-semibold text-gray-800">AI 助手</h3>
                            <p className="text-xs text-gray-500">{projectContext?.projectName ?? "工程管家"}</p>
                        </div>
                        <button
                            onClick={resetSession}
                            className="text-xs text-gray-400 active:text-gray-600"
                        >
                            新对话
                        </button>
                    </div>

                    {/* 消息列表 */}
                    <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-1">
                        {messages.length === 0 && (
                            <div className="text-center text-gray-400 text-sm py-8">
                                <p>您好！我是 AI 助手。</p>
                                <p className="mt-1">试试说「今天焊了80米」</p>
                            </div>
                        )}
                        {messages.filter((m) => m.role !== "system").map((msg, i) => (
                            <AiChatBubble key={i} role={msg.role as "user" | "assistant"} content={msg.content} />
                        ))}
                        {loading && (
                            <div className="flex justify-start mb-2">
                                <div className="bg-gray-100 rounded-lg px-3 py-2 text-sm text-gray-500">
                                    思考中...
                                </div>
                            </div>
                        )}
                    </div>

                    {/* 快捷操作 */}
                    {messages.length === 0 && (
                        <AiChatQuickActions
                            actions={QUICK_ACTIONS.map((a) => ({
                                module: a.module,
                                icon: a.icon,
                                disabled: filledModules.has(a.module),
                                onClick: () => handleQuickAction(a.module),
                            }))}
                            onFillAll={handleFillAll}
                            allDisabled={filledModules.size >= 4}
                        />
                    )}

                    {/* 确认按钮 */}
                    {lastIntent && (
                        <div className="border-t px-4 py-2 flex gap-2">
                            <button
                                onClick={confirmSubmit}
                                className="flex-1 bg-eng-green text-white rounded-lg py-2 text-sm font-medium active:opacity-80"
                            >
                                确认提交
                            </button>
                            <button
                                onClick={() => sendMessage("取消")}
                                className="flex-1 bg-gray-100 text-gray-600 rounded-lg py-2 text-sm active:bg-gray-200"
                            >
                                取消
                            </button>
                        </div>
                    )}

                    {/* 输入框 */}
                    <div className="border-t px-4 py-3 flex gap-2">
                        <input
                            ref={inputRef}
                            type="text"
                            placeholder="输入消息..."
                            className="flex-1 rounded-lg border border-gray-200 px-3 py-2 text-sm focus:outline-none focus:border-eng-blue"
                            onKeyDown={handleKeyDown}
                            disabled={loading}
                        />
                        <button
                            onClick={handleSend}
                            disabled={loading}
                            className="rounded-lg bg-eng-blue px-4 py-2 text-sm text-white active:opacity-80 disabled:opacity-50"
                        >
                            发送
                        </button>
                    </div>
                </div>
            </div>
        </>
    );

    return createPortal(widget, document.body);
}
