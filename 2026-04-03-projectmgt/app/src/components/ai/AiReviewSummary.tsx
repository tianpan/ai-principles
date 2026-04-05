"use client";

import React from "react";
import type { PreReviewResult } from "../../lib/ai/smart-review";

interface AiReviewSummaryProps {
    result: PreReviewResult;
    onBatchApprove: () => void;
}

export function AiReviewSummary({ result, onBatchApprove }: AiReviewSummaryProps) {
    const total = result.normal.length + result.suspicious.length + result.critical.length;
    if (total === 0) return null;

    return (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-800">AI 预审摘要</h3>
                <span className="text-xs text-gray-400">共 {total} 条</span>
            </div>

            <div className="grid grid-cols-3 gap-3 mb-3">
                {/* 正常 */}
                <div className="rounded-lg bg-green-50 p-3 text-center">
                    <div className="text-2xl font-bold text-eng-green">
                        {result.normal.length}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">正常</div>
                </div>
                {/* 疑似异常 */}
                <div className="rounded-lg bg-yellow-50 p-3 text-center">
                    <div className="text-2xl font-bold text-eng-orange">
                        {result.suspicious.length}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">疑似异常</div>
                </div>
                {/* 明显问题 */}
                <div className="rounded-lg bg-red-50 p-3 text-center">
                    <div className="text-2xl font-bold text-red-600">
                        {result.critical.length}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">明显问题</div>
                </div>
            </div>

            {result.normal.length > 0 && (
                <button
                    onClick={onBatchApprove}
                    className="w-full rounded-lg bg-eng-green text-white py-2 text-sm font-medium hover:opacity-90 transition-opacity"
                >
                    一键通过 {result.normal.length} 条正常记录
                </button>
            )}

            {result.suspicious.length > 0 && (
                <div className="mt-3 space-y-2">
                    <div className="text-xs font-medium text-eng-orange">疑似异常：</div>
                    {result.suspicious.map(({ record, results }) => (
                        <div key={record.id} className="rounded border border-yellow-200 bg-yellow-50 p-2 text-xs">
                            <span className="font-medium">记录 #{record.id}</span>
                            <span className="text-gray-500 ml-2">({record.moduleType} · {record.date})</span>
                            <div className="mt-1 text-gray-600">
                                {results.map((r) => r.reason).join("；")}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {result.critical.length > 0 && (
                <div className="mt-3 space-y-2">
                    <div className="text-xs font-medium text-red-600">明显问题：</div>
                    {result.critical.map(({ record, results }) => (
                        <div key={record.id} className="rounded border border-red-200 bg-red-50 p-2 text-xs">
                            <span className="font-medium">记录 #{record.id}</span>
                            <span className="text-gray-500 ml-2">({record.moduleType} · {record.date})</span>
                            <div className="mt-1 text-gray-600">
                                {results.map((r) => r.reason).join("；")}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
