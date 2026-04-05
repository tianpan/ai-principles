"use client";

import React from "react";

interface Risk {
    type: string;
    severity: "高" | "中" | "低";
    description: string;
}

interface AiBriefingCardProps {
    content: string;
    risks: Risk[];
    date: string;
    projectName: string;
    compact?: boolean;
}

const severityColors: Record<string, string> = {
    高: "bg-red-100 text-red-700",
    中: "bg-yellow-100 text-yellow-700",
    低: "bg-gray-100 text-gray-600",
};

export function AiBriefingCard({
    content,
    risks,
    date,
    projectName,
    compact = false,
}: AiBriefingCardProps) {
    if (compact) {
        return (
            <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-gray-700">
                        今日简报 · {projectName}
                    </span>
                    <span className="text-xs text-gray-400">{date}</span>
                </div>
                {risks.length > 0 ? (
                    <div className="flex gap-1 flex-wrap">
                        {risks.map((risk, i) => (
                            <span
                                key={i}
                                className={`text-xs px-1.5 py-0.5 rounded ${severityColors[risk.severity]}`}
                            >
                                {risk.type}
                            </span>
                        ))}
                    </div>
                ) : (
                    <p className="text-xs text-gray-500">施工进展正常，无风险提示</p>
                )}
            </div>
        );
    }

    return (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-800">
                    AI 每日简报 · {projectName}
                </h3>
                <span className="text-xs text-gray-400">{date}</span>
            </div>

            {/* Markdown-like content */}
            <div className="text-xs text-gray-600 whitespace-pre-line leading-relaxed mb-3">
                {content.split("\n").map((line, i) => {
                    if (line.startsWith("# "))
                        return <h4 key={i} className="text-sm font-bold text-gray-800 mt-2">{line.slice(2)}</h4>;
                    if (line.startsWith("## "))
                        return <h5 key={i} className="text-xs font-semibold text-gray-700 mt-2 mb-1">{line.slice(3)}</h5>;
                    if (line.startsWith("**") && line.endsWith("**"))
                        return <p key={i} className="font-medium text-gray-700">{line.slice(2, -2)}</p>;
                    if (line.startsWith("- "))
                        return <p key={i} className="pl-2">• {line.slice(2)}</p>;
                    if (line.startsWith("---"))
                        return <hr key={i} className="border-gray-100 my-2" />;
                    if (line.startsWith("*") && line.endsWith("*"))
                        return <p key={i} className="text-gray-400 italic">{line.slice(1, -1)}</p>;
                    return <p key={i}>{line}</p>;
                })}
            </div>

            {/* Risk badges */}
            {risks.length > 0 && (
                <div className="border-t border-gray-100 pt-2 space-y-1">
                    <span className="text-xs font-medium text-gray-500">风险提示：</span>
                    {risks.map((risk, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                            <span className={`px-1.5 py-0.5 rounded font-medium ${severityColors[risk.severity]}`}>
                                {risk.severity}
                            </span>
                            <span className="text-gray-600">{risk.description}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
