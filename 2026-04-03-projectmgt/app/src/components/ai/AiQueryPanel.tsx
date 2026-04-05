"use client";

import React from "react";
import type { NLQueryResult } from "../../lib/ai/nl-query";

interface AiQueryPanelProps {
    result: NLQueryResult;
    onClose: () => void;
}

const queryTypeColors: Record<string, string> = {
    进度: "bg-green-50 text-eng-green",
    材料: "bg-orange-50 text-eng-orange",
    考勤: "bg-blue-50 text-eng-blue",
    综合: "bg-gray-50 text-gray-600",
};

export function AiQueryPanel({ result, onClose }: AiQueryPanelProps) {
    return (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-lg">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <span
                        className={`text-xs px-2 py-0.5 rounded-full font-medium ${queryTypeColors[result.queryType]}`}
                    >
                        {result.queryType}查询
                    </span>
                    <span className="text-xs text-gray-400">
                        {result.question}
                    </span>
                </div>
                <button
                    onClick={onClose}
                    className="text-gray-400 hover:text-gray-600 text-sm"
                >
                    ✕
                </button>
            </div>

            <p className="text-sm text-gray-700 whitespace-pre-line leading-relaxed mb-3">
                {result.answer}
            </p>

            {result.data.length > 0 && (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {result.data.map((item, i) => (
                        <div
                            key={i}
                            className="rounded-lg bg-gray-50 p-2 text-center"
                        >
                            <p className="text-xs text-gray-500">{item.label}</p>
                            <p className="text-sm font-medium text-gray-800 mt-0.5">
                                {item.value}
                            </p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
