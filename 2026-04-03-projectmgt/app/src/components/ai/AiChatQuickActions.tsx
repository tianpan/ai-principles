"use client";

import React from "react";

interface QuickAction {
    module: string;
    icon: string;
    disabled: boolean;
    onClick: () => void;
}

interface AiChatQuickActionsProps {
    actions: QuickAction[];
    onFillAll: () => void;
    allDisabled: boolean;
}

export function AiChatQuickActions({
    actions,
    onFillAll,
    allDisabled,
}: AiChatQuickActionsProps) {
    return (
        <div className="border-t border-gray-200 bg-gray-50 p-3">
            <div className="text-xs text-gray-500 mb-2">快捷操作</div>
            <div className="grid grid-cols-2 gap-2 mb-2">
                {actions.map((action) => (
                    <button
                        key={action.module}
                        onClick={action.onClick}
                        disabled={action.disabled}
                        className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors ${
                            action.disabled
                                ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                                : "bg-white border border-gray-200 text-gray-700 active:bg-gray-100"
                        }`}
                    >
                        <span>{action.icon}</span>
                        <span>{action.module}</span>
                    </button>
                ))}
            </div>
            <button
                onClick={onFillAll}
                disabled={allDisabled}
                className={`w-full rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                    allDisabled
                        ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                        : "bg-eng-blue text-white active:bg-blue-700"
                }`}
            >
                全部帮我填
            </button>
        </div>
    );
}
