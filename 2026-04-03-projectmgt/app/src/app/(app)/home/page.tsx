"use client";

import { useState, useMemo } from "react";
import { useMockData } from "@/lib/MockDataProvider";
import { ProjectSelector } from "@/components/ProjectSelector";
import { AiBriefingCard } from "@/components/ai/AiBriefingCard";
import Link from "next/link";

const modules = [
    {
        key: "考勤",
        label: "安全考勤",
        href: "/safety",
        icon: "🛡️",
        color: "bg-blue-50 text-eng-blue",
    },
    {
        key: "进度",
        label: "施工进度",
        href: "/progress",
        icon: "📊",
        color: "bg-green-50 text-eng-green",
    },
    {
        key: "材料",
        label: "材料设备",
        href: "/material",
        icon: "📦",
        color: "bg-orange-50 text-eng-orange",
    },
    {
        key: "施工记录",
        label: "施工记录",
        href: "/site-record",
        icon: "📝",
        color: "bg-purple-50 text-purple-600",
    },
];

export default function HomePage() {
    const { currentUser, getProjectsByTeamLeader, getDailyRecordsByProject, data, getAiBriefingByProjectAndDate } =
        useMockData();

    const isTeamLeader = currentUser?.role === "班组长";
    const myProjects = currentUser
        ? getProjectsByTeamLeader(currentUser.id)
        : [];
    const autoProject = myProjects[0];
    const [selectedProjectId, setSelectedProjectId] = useState<number | null>(
        autoProject?.id ?? null
    );
    const currentProject = selectedProjectId
        ? data.projects.find((p) => p.id === selectedProjectId)
        : autoProject;

    const today = new Date().toISOString().split("T")[0];

    // AI 简报 — 仅项目经理可见
    const isProjectManager = currentUser?.role === "金卓项目经理";
    const briefing = useMemo(() => {
        if (!isProjectManager || !currentProject) return null;
        return getAiBriefingByProjectAndDate(currentProject.id, today);
    }, [isProjectManager, currentProject, today, getAiBriefingByProjectAndDate]);

    if (!currentUser) return null;
    const todayRecords = currentProject
        ? getDailyRecordsByProject(currentProject.id).filter(
              (r) => r.date === today
          )
        : [];

    const completedModules = new Set(todayRecords.map((r) => r.moduleType));

    const pendingRecords = data.dailyRecords.filter(
        (r) =>
            currentProject &&
            r.projectId === currentProject.id &&
            r.status === "待审核"
    );

    return (
        <div className="px-4 py-4 space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <p className="text-sm text-eng-gray-400">
                        {today}{" "}
                        {getDayOfWeek(today)}
                    </p>
                    <h2 className="text-app-title font-bold text-eng-gray-900">
                        你好，{currentUser.name}
                    </h2>
                </div>
                <div className="w-9 h-9 bg-eng-blue rounded-full flex items-center justify-center text-white text-sm font-medium">
                    {currentUser.name[0]}
                </div>
            </div>

            {/* Project selector for non-班组长 */}
            {!isTeamLeader && (
                <ProjectSelector
                    value={selectedProjectId}
                    onChange={(id) => setSelectedProjectId(id)}
                    className="w-full"
                />
            )}

            {/* Current Project */}
            {currentProject && (
                <div className="bg-white rounded-card border border-eng-gray-100 p-4 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium text-eng-gray-900 text-sm">
                            {currentProject.name}
                        </h3>
                        <span className="text-xs px-2 py-0.5 bg-eng-blue/10 text-eng-blue rounded-full">
                            {currentProject.status}
                        </span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-eng-gray-400">
                        <span>
                            {currentProject.startDate} ~{" "}
                            {currentProject.endDate}
                        </span>
                        <span>
                            {currentProject.workItems.length} 项工序
                        </span>
                    </div>
                </div>
            )}

            {!currentProject && (
                <div className="py-8 text-center">
                    <p className="text-sm text-gray-400">
                        {isTeamLeader ? "暂无负责的项目" : "请选择一个项目"}
                    </p>
                </div>
            )}

            {/* AI 简报摘要 — 项目经理可见 */}
            {briefing && isProjectManager && (
                <AiBriefingCard
                    content={briefing.content}
                    risks={briefing.risks}
                    date={briefing.date}
                    projectName={currentProject?.name ?? ""}
                    compact
                />
            )}

            {/* Module Cards */}
            <div>
                <h3 className="text-sm font-medium text-eng-gray-700 mb-2">
                    今日登记
                </h3>
                <div className="grid grid-cols-2 gap-3">
                    {modules.map((mod) => {
                        const done = completedModules.has(mod.key);
                        return (
                            <Link
                                key={mod.key}
                                href={mod.href}
                                className={`relative p-4 rounded-card border transition-shadow hover:shadow-md ${
                                    done
                                        ? "border-eng-green/30 bg-eng-green/5"
                                        : "border-eng-gray-100 bg-white"
                                }`}
                            >
                                <div className="text-2xl mb-2">{mod.icon}</div>
                                <p className="text-sm font-medium text-eng-gray-800">
                                    {mod.label}
                                </p>
                                {done && (
                                    <span className="absolute top-2 right-2 text-eng-green text-xs">
                                        ✓ 已填
                                    </span>
                                )}
                            </Link>
                        );
                    })}
                </div>
            </div>

            {/* Recent Pending Reviews */}
            {pendingRecords.length > 0 && (
                <div>
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-medium text-eng-gray-700">
                            待审核
                        </h3>
                        <Link
                            href="/review"
                            className="text-xs text-eng-blue"
                        >
                            查看全部
                        </Link>
                    </div>
                    <div className="space-y-2">
                        {pendingRecords.slice(0, 3).map((record) => (
                            <div
                                key={record.id}
                                className="flex items-center justify-between p-3 bg-white rounded-card border border-eng-gray-100"
                            >
                                <div>
                                    <p className="text-sm text-eng-gray-800">
                                        {record.moduleType} · {record.date}
                                    </p>
                                    <p className="text-xs text-eng-gray-400">
                                        {currentProject?.name}
                                    </p>
                                </div>
                                <span className="text-xs px-2 py-0.5 bg-orange-50 text-eng-orange rounded-full">
                                    待审核
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

function getDayOfWeek(dateStr: string): string {
    const days = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
    return days[new Date(dateStr).getDay()];
}
