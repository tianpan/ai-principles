"use client";

import { useState, useMemo } from "react";
import { useMockData } from "@/lib/MockDataProvider";

const moduleFilters = ["全部", "考勤", "进度", "材料", "施工记录"];

const statusLabels: Record<string, { text: string; cls: string }> = {
    已通过: { text: "已通过", cls: "bg-green-50 text-eng-green" },
    待审核: { text: "待审核", cls: "bg-orange-50 text-eng-orange" },
    已退回: { text: "已退回", cls: "bg-red-50 text-eng-red" },
};

export default function HistoryPage() {
    const { currentUser, getProjectsByTeamLeader, getDailyRecordsByProject } =
        useMockData();
    const [filter, setFilter] = useState("全部");

    const allRecords = useMemo(() => {
        if (!currentUser) return [];
        const myProjects = getProjectsByTeamLeader(currentUser.id);
        return myProjects.flatMap((p) =>
            getDailyRecordsByProject(p.id).map((r) => ({
                ...r,
                projectName: p.name,
            }))
        );
    }, [currentUser, getProjectsByTeamLeader, getDailyRecordsByProject]);

    const filtered = useMemo(
        () =>
            filter === "全部"
                ? allRecords
                : allRecords.filter((r) => r.moduleType === filter),
        [filter, allRecords]
    );

    const grouped = useMemo(() => {
        const map = new Map<string, typeof filtered>();
        for (const r of filtered) {
            const existing = map.get(r.date) || [];
            existing.push(r);
            map.set(r.date, existing);
        }
        return Array.from(map.entries()).sort(
            (a, b) => b[0].localeCompare(a[0])
        );
    }, [filtered]);

    if (!currentUser) return null;

    return (
        <div className="px-4 py-4 space-y-4">
            <h1 className="text-app-title font-bold text-eng-gray-900">
                历史记录
            </h1>

            {/* Module Filter */}
            <div className="flex gap-2 overflow-x-auto pb-1">
                {moduleFilters.map((m) => (
                    <button
                        key={m}
                        onClick={() => setFilter(m)}
                        className={`shrink-0 px-3 py-1.5 rounded-full text-xs transition-colors ${
                            filter === m
                                ? "bg-eng-blue text-white"
                                : "bg-eng-gray-100 text-eng-gray-600"
                        }`}
                    >
                        {m}
                    </button>
                ))}
            </div>

            {/* Grouped Records */}
            {grouped.map(([date, records]) => (
                <div key={date}>
                    <h3 className="text-xs text-eng-gray-400 mb-2">
                        {date} {getDayOfWeek(date)}
                    </h3>
                    <div className="space-y-2">
                        {records.map((r) => {
                            const st = statusLabels[r.status] || {
                                text: r.status,
                                cls: "bg-gray-50 text-gray-500",
                            };
                            return (
                                <div
                                    key={r.id}
                                    className="flex items-center justify-between p-3 bg-white rounded-card border border-eng-gray-100"
                                >
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm text-eng-gray-800 font-medium">
                                            {r.moduleType}
                                        </p>
                                        <p className="text-xs text-eng-gray-400 truncate">
                                            {r.projectName}
                                        </p>
                                    </div>
                                    <span
                                        className={`shrink-0 text-xs px-2 py-0.5 rounded-full ml-2 ${st.cls}`}
                                    >
                                        {st.text}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            ))}

            {grouped.length === 0 && (
                <p className="text-sm text-eng-gray-400 text-center py-8">
                    暂无记录
                </p>
            )}
        </div>
    );
}

function getDayOfWeek(dateStr: string): string {
    const days = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
    return days[new Date(dateStr).getDay()];
}
