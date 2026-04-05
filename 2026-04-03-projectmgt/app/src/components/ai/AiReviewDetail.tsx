"use client";

import React from "react";
import type { AuditResult } from "../../lib/ai/rule-engine";

interface AiReviewDetailProps {
    recordId: number;
    moduleType: string;
    date: string;
    results: AuditResult[];
    onApprove: () => void;
    onReject: () => void;
}

export function AiReviewDetail({
    recordId,
    moduleType,
    date,
    results,
    onApprove,
    onReject,
}: AiReviewDetailProps) {
    return (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-semibold text-gray-800">
                    记录 #{recordId} · {moduleType}
                </h4>
                <span className="text-xs text-gray-400">{date}</span>
            </div>

            <div className="space-y-2 mb-4">
                {results.map((result, i) => (
                    <div
                        key={i}
                        className={`rounded-lg p-3 text-sm ${
                            result.severity === "明显问题"
                                ? "bg-red-50 border border-red-200"
                                : result.severity === "疑似异常"
                                ? "bg-yellow-50 border border-yellow-200"
                                : "bg-gray-50 border border-gray-200"
                        }`}
                    >
                        <div className="flex items-center gap-2 mb-1">
                            <span
                                className={`inline-block rounded px-1.5 py-0.5 text-xs font-medium ${
                                    result.severity === "明显问题"
                                        ? "bg-red-100 text-red-700"
                                        : result.severity === "疑似异常"
                                        ? "bg-yellow-100 text-yellow-700"
                                        : "bg-gray-100 text-gray-700"
                                }`}
                            >
                                {result.severity}
                            </span>
                            <span className="font-medium text-gray-700">
                                {result.ruleName}
                            </span>
                        </div>
                        <p className="text-gray-600 text-xs">{result.reason}</p>
                        <p className="text-gray-500 text-xs mt-1">
                            建议：{result.suggestion}
                        </p>
                    </div>
                ))}
            </div>

            <div className="flex gap-2">
                <button
                    onClick={onApprove}
                    className="flex-1 rounded-lg bg-eng-green text-white py-2 text-sm font-medium hover:opacity-90"
                >
                    通过
                </button>
                <button
                    onClick={onReject}
                    className="flex-1 rounded-lg bg-red-500 text-white py-2 text-sm font-medium hover:opacity-90"
                >
                    退回
                </button>
            </div>
        </div>
    );
}
