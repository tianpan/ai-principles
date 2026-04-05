"use client";

import { useState, useMemo } from "react";
import { useMockData } from "@/lib/MockDataProvider";

export default function PcReportsPage() {
    const { currentUser, data, getCumulativeProgress } = useMockData();
    const [selectedProject, setSelectedProject] = useState<number | null>(null);
    const [reportTab, setReportTab] = useState<"progress" | "safety" | "completion">("progress");

    const projects = selectedProject
        ? data.projects.filter((p) => p.id === selectedProject)
        : data.projects;

    // Progress data per project
    const progressData = useMemo(() => {
        return projects.map((project) => {
            const workItemProgress = project.workItems.map((wi) => {
                const cumulative = getCumulativeProgress(project.id, wi.id);
                const pct = Math.min(cumulative / wi.targetQuantity, 1);
                return {
                    ...wi,
                    cumulative,
                    percent: pct,
                };
            });
            const overallProgress = workItemProgress.reduce(
                (sum, wi) => sum + wi.percent * wi.weight,
                0
            );
            const daysLeft = Math.ceil(
                (new Date(project.endDate).getTime() - Date.now()) /
                    (1000 * 60 * 60 * 24)
            );
            const leader = data.users.find(
                (u) => u.id === project.teamLeaderId
            );
            return {
                project,
                leader,
                workItemProgress,
                overallProgress,
                daysLeft,
            };
        });
    }, [projects, data.users, getCumulativeProgress]);

    // Safety summary
    const safetySummary = useMemo(() => {
        const approvedRecords = data.dailyRecords.filter(
            (r) => r.moduleType === "考勤" && r.status === "已通过"
        );
        const totalBriefings = approvedRecords.length;
        const byType: Record<string, number> = {};
        for (const r of approvedRecords) {
            const type = (r.attendance as Record<string, unknown>)
                ?.safetyBriefingType as string;
            if (type) {
                byType[type] = (byType[type] || 0) + 1;
            }
        }
        // Count unique dates with safety records
        const uniqueDates = new Set(approvedRecords.map((r) => r.date));
        return {
            totalBriefings,
            uniqueDays: uniqueDates.size,
            byType,
        };
    }, [data.dailyRecords]);

    // Completion submission tracker
    const completionTracker = useMemo(() => {
        const today = new Date().toISOString().split("T")[0];
        return projects.map((project) => {
            const leader = data.users.find(
                (u) => u.id === project.teamLeaderId
            );
            const todayRecords = data.dailyRecords.filter(
                (r) =>
                    r.projectId === project.id &&
                    r.date === today
            );
            const modulesSubmitted = new Set(todayRecords.map((r) => r.moduleType));
            const pendingCount = todayRecords.filter(
                (r) => r.status === "待审核"
            ).length;
            const approvedCount = todayRecords.filter(
                (r) => r.status === "已通过"
            ).length;
            const returnedCount = todayRecords.filter(
                (r) => r.status === "已退回"
            ).length;

            return {
                project,
                leader,
                modulesSubmitted,
                moduleCount: modulesSubmitted.size,
                pendingCount,
                approvedCount,
                returnedCount,
                totalRecords: todayRecords.length,
            };
        });
    }, [projects, data.dailyRecords, data.users]);

    const tabs = [
        { key: "progress" as const, label: "进度总览" },
        { key: "safety" as const, label: "安全汇总" },
        { key: "completion" as const, label: "提交追踪" },
    ];

    if (!currentUser) return null;

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <h1 className="text-xl font-bold text-gray-800">
                    报表中心
                </h1>
                <select
                    value={selectedProject ?? ""}
                    onChange={(e) =>
                        setSelectedProject(
                            e.target.value ? Number(e.target.value) : null
                        )
                    }
                    className="h-9 px-3 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                >
                    <option value="">全部项目</option>
                    {data.projects.map((p) => (
                        <option key={p.id} value={p.id}>
                            {p.name}
                        </option>
                    ))}
                </select>
            </div>

            {/* Tab Switcher */}
            <div className="flex gap-0 border-b border-gray-200">
                {tabs.map((tab) => (
                    <button
                        key={tab.key}
                        onClick={() => setReportTab(tab.key)}
                        className={`px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                            reportTab === tab.key
                                ? "border-eng-blue text-eng-blue"
                                : "border-transparent text-gray-500 hover:text-gray-700"
                        }`}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Progress Overview */}
            {reportTab === "progress" && (
                <div className="space-y-4">
                    {/* Summary Cards */}
                    <div className="grid grid-cols-4 gap-4">
                        <SummaryCard
                            label="项目总数"
                            value={progressData.length}
                            color="text-eng-blue"
                            bg="bg-blue-50"
                        />
                        <SummaryCard
                            label="平均进度"
                            value={`${(progressData.reduce((s, p) => s + p.overallProgress, 0) / Math.max(progressData.length, 1) * 100).toFixed(1)}%`}
                            color="text-eng-green"
                            bg="bg-green-50"
                        />
                        <SummaryCard
                            label="即将到期"
                            value={progressData.filter((p) => p.daysLeft >= 0 && p.daysLeft <= 7).length}
                            color="text-eng-orange"
                            bg="bg-orange-50"
                        />
                        <SummaryCard
                            label="已超期"
                            value={progressData.filter((p) => p.daysLeft < 0).length}
                            color="text-eng-red"
                            bg="bg-red-50"
                        />
                    </div>

                    {/* Detailed Progress Table */}
                    {progressData.map((pd) => (
                        <div
                            key={pd.project.id}
                            className="bg-white rounded-lg border border-gray-200 overflow-hidden"
                        >
                            <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-b border-gray-100">
                                <div className="flex items-center gap-3">
                                    <span className="text-sm font-medium text-gray-800">
                                        {pd.project.name}
                                    </span>
                                    <span className="text-xs text-gray-400">
                                        {pd.leader?.name}
                                    </span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <span
                                        className={`text-xs font-medium ${
                                            pd.daysLeft < 0
                                                ? "text-eng-red"
                                                : pd.daysLeft <= 7
                                                ? "text-eng-orange"
                                                : "text-eng-green"
                                        }`}
                                    >
                                        {pd.daysLeft > 0
                                            ? `剩余 ${pd.daysLeft} 天`
                                            : pd.daysLeft === 0
                                            ? "今日截止"
                                            : `超期 ${Math.abs(pd.daysLeft)} 天`}
                                    </span>
                                    <span className="text-sm font-bold text-eng-blue">
                                        {(pd.overallProgress * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>

                            {/* Overall Progress Bar */}
                            <div className="px-4 py-3 border-b border-gray-50">
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-gray-500 w-16">
                                        综合进度
                                    </span>
                                    <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-eng-blue rounded-full transition-all"
                                            style={{
                                                width: `${Math.min(pd.overallProgress * 100, 100)}%`,
                                            }}
                                        />
                                    </div>
                                </div>
                            </div>

                            {/* Work Item Breakdown */}
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-gray-100 bg-gray-50/50">
                                        <th className="text-left px-4 py-2 text-gray-500 font-medium text-xs">
                                            工序
                                        </th>
                                        <th className="text-right px-4 py-2 text-gray-500 font-medium text-xs">
                                            计划量
                                        </th>
                                        <th className="text-right px-4 py-2 text-gray-500 font-medium text-xs">
                                            累计完成
                                        </th>
                                        <th className="text-right px-4 py-2 text-gray-500 font-medium text-xs">
                                            偏差
                                        </th>
                                        <th className="text-left px-4 py-2 text-gray-500 font-medium text-xs w-40">
                                            进度
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {pd.workItemProgress.map((wi) => {
                                        const deviation =
                                            wi.cumulative -
                                            wi.targetQuantity *
                                                (pd.daysLeft > 0
                                                    ? 1 -
                                                      pd.daysLeft /
                                                          Math.ceil(
                                                              (new Date(
                                                                  pd.project.endDate
                                                              ).getTime() -
                                                                  new Date(
                                                                      pd.project.startDate
                                                                  ).getTime()) /
                                                                  (1000 *
                                                                      60 *
                                                                      60 *
                                                                      24)
                                                          )
                                                    : 1);
                                        return (
                                            <tr
                                                key={wi.id}
                                                className="border-b border-gray-50"
                                            >
                                                <td className="px-4 py-2.5 text-gray-700 text-xs">
                                                    {wi.name}
                                                    <span className="ml-1 text-gray-400">
                                                        ({wi.weight}%)
                                                    </span>
                                                </td>
                                                <td className="px-4 py-2.5 text-right text-xs text-gray-600">
                                                    {wi.targetQuantity}{" "}
                                                    {wi.unit}
                                                </td>
                                                <td className="px-4 py-2.5 text-right text-xs text-gray-600">
                                                    {wi.cumulative.toFixed(1)}{" "}
                                                    {wi.unit}
                                                </td>
                                                <td
                                                    className={`px-4 py-2.5 text-right text-xs font-medium ${
                                                        deviation >= 0
                                                            ? "text-eng-green"
                                                            : "text-eng-red"
                                                    }`}
                                                >
                                                    {deviation >= 0 ? "+" : ""}
                                                    {deviation.toFixed(1)}
                                                </td>
                                                <td className="px-4 py-2.5">
                                                    <div className="flex items-center gap-2">
                                                        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                                            <div
                                                                className={`h-full rounded-full ${
                                                                    wi.percent >=
                                                                    0.9
                                                                        ? "bg-eng-green"
                                                                        : wi.percent >=
                                                                          0.6
                                                                        ? "bg-eng-blue"
                                                                        : wi.percent >=
                                                                          0.3
                                                                        ? "bg-eng-orange"
                                                                        : "bg-eng-red"
                                                                }`}
                                                                style={{
                                                                    width: `${Math.min(wi.percent * 100, 100)}%`,
                                                                }}
                                                            />
                                                        </div>
                                                        <span className="text-xs text-gray-500 w-10 text-right">
                                                            {(
                                                                wi.percent * 100
                                                            ).toFixed(0)}
                                                            %
                                                        </span>
                                                    </div>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    ))}
                </div>
            )}

            {/* Safety Summary */}
            {reportTab === "safety" && (
                <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                        <div className="bg-white rounded-lg border border-gray-200 p-6">
                            <p className="text-sm text-gray-500">
                                安全交底总次数
                            </p>
                            <p className="text-3xl font-bold text-eng-blue mt-1">
                                {safetySummary.totalBriefings}
                            </p>
                            <p className="text-xs text-gray-400 mt-1">
                                覆盖 {safetySummary.uniqueDays} 个工作日
                            </p>
                        </div>
                        <div className="bg-white rounded-lg border border-gray-200 p-6">
                            <p className="text-sm text-gray-500">
                                交底类型分布
                            </p>
                            <div className="mt-2 space-y-2">
                                {Object.entries(safetySummary.byType).map(
                                    ([type, count]) => (
                                        <div
                                            key={type}
                                            className="flex items-center gap-2"
                                        >
                                            <span className="text-xs text-gray-600 w-24 truncate">
                                                {type}
                                            </span>
                                            <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-eng-blue rounded-full"
                                                    style={{
                                                        width: `${(count / Math.max(safetySummary.totalBriefings, 1)) * 100}%`,
                                                    }}
                                                />
                                            </div>
                                            <span className="text-xs text-gray-500 w-8 text-right">
                                                {count}
                                            </span>
                                        </div>
                                    )
                                )}
                                {Object.keys(safetySummary.byType).length ===
                                    0 && (
                                    <p className="text-xs text-gray-400">
                                        暂无数据
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Monthly Trend (Mock) */}
                    <div className="bg-white rounded-lg border border-gray-200 p-6">
                        <h3 className="text-sm font-medium text-gray-700 mb-4">
                            安全交底月度趋势
                        </h3>
                        <div className="flex items-end gap-3 h-32">
                            {["1月", "2月", "3月", "4月"].map(
                                (month, i) => {
                                    const heights = [45, 60, 75, safetySummary.totalBriefings];
                                    const h = Math.max((heights[i] / 100) * 100, 8);
                                    return (
                                        <div
                                            key={month}
                                            className="flex-1 flex flex-col items-center gap-1"
                                        >
                                            <span className="text-xs text-gray-500">
                                                {heights[i]}
                                            </span>
                                            <div
                                                className={`w-full rounded-t ${
                                                    i === 3
                                                        ? "bg-eng-blue"
                                                        : "bg-blue-100"
                                                }`}
                                                style={{ height: `${h}%` }}
                                            />
                                            <span className="text-xs text-gray-400">
                                                {month}
                                            </span>
                                        </div>
                                    );
                                }
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Completion Tracker */}
            {reportTab === "completion" && (
                <div className="space-y-4">
                    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-gray-100 bg-gray-50">
                                    <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                        项目名称
                                    </th>
                                    <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                        班组长
                                    </th>
                                    <th className="text-center px-4 py-3 text-gray-500 font-medium">
                                        今日提交
                                    </th>
                                    <th className="text-center px-4 py-3 text-gray-500 font-medium">
                                        待审核
                                    </th>
                                    <th className="text-center px-4 py-3 text-gray-500 font-medium">
                                        已通过
                                    </th>
                                    <th className="text-center px-4 py-3 text-gray-500 font-medium">
                                        已退回
                                    </th>
                                    <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                        模块覆盖
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                {completionTracker.map((ct) => (
                                    <tr
                                        key={ct.project.id}
                                        className="border-b border-gray-50 hover:bg-gray-50"
                                    >
                                        <td className="px-4 py-3 text-gray-800 font-medium">
                                            {ct.project.name}
                                        </td>
                                        <td className="px-4 py-3 text-gray-600">
                                            {ct.leader?.name || "-"}
                                        </td>
                                        <td className="px-4 py-3 text-center">
                                            <span className="text-sm font-medium text-gray-700">
                                                {ct.totalRecords}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 text-center">
                                            <span
                                                className={`text-sm font-medium ${
                                                    ct.pendingCount > 0
                                                        ? "text-eng-orange"
                                                        : "text-gray-300"
                                                }`}
                                            >
                                                {ct.pendingCount}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 text-center">
                                            <span className="text-sm font-medium text-eng-green">
                                                {ct.approvedCount}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 text-center">
                                            <span
                                                className={`text-sm font-medium ${
                                                    ct.returnedCount > 0
                                                        ? "text-eng-red"
                                                        : "text-gray-300"
                                                }`}
                                            >
                                                {ct.returnedCount}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <ModuleBadges
                                                modules={ct.modulesSubmitted}
                                            />
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Deadline Countdown */}
                    <div className="bg-white rounded-lg border border-gray-200 p-6">
                        <h3 className="text-sm font-medium text-gray-700 mb-4">
                            项目截止倒计时
                        </h3>
                        <div className="space-y-3">
                            {progressData
                                .sort((a, b) => a.daysLeft - b.daysLeft)
                                .map((pd) => (
                                    <div
                                        key={pd.project.id}
                                        className="flex items-center gap-4"
                                    >
                                        <span className="text-sm text-gray-700 w-40 truncate">
                                            {pd.project.name}
                                        </span>
                                        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                            <div
                                                className={`h-full rounded-full ${
                                                    pd.daysLeft < 0
                                                        ? "bg-eng-red"
                                                        : pd.daysLeft <= 7
                                                        ? "bg-eng-orange"
                                                        : "bg-eng-blue"
                                                }`}
                                                style={{
                                                    width: `${Math.max(Math.min((1 - pd.daysLeft / Math.ceil((new Date(pd.project.endDate).getTime() - new Date(pd.project.startDate).getTime()) / (1000 * 60 * 60 * 24))) * 100, 100), 5)}%`,
                                                }}
                                            />
                                        </div>
                                        <span
                                            className={`text-xs font-medium w-20 text-right ${
                                                pd.daysLeft < 0
                                                    ? "text-eng-red"
                                                    : pd.daysLeft <= 7
                                                    ? "text-eng-orange"
                                                    : "text-eng-green"
                                            }`}
                                        >
                                            {pd.daysLeft > 0
                                                ? `${pd.daysLeft} 天`
                                                : pd.daysLeft === 0
                                                ? "今日截止"
                                                : `超期 ${Math.abs(pd.daysLeft)} 天`}
                                        </span>
                                    </div>
                                ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

function SummaryCard({
    label,
    value,
    color,
    bg,
}: {
    label: string;
    value: string | number;
    color: string;
    bg: string;
}) {
    return (
        <div className={`${bg} rounded-lg p-4`}>
            <p className="text-sm text-gray-500">{label}</p>
            <p className={`text-2xl font-bold ${color} mt-1`}>{value}</p>
        </div>
    );
}

function ModuleBadges({
    modules,
}: {
    modules: Set<string>;
}) {
    const allModules = ["考勤", "进度", "材料", "施工记录"];
    return (
        <div className="flex gap-1">
            {allModules.map((m) => (
                <span
                    key={m}
                    className={`text-xs px-1.5 py-0.5 rounded ${
                        modules.has(m)
                            ? "bg-blue-50 text-eng-blue"
                            : "bg-gray-50 text-gray-300"
                    }`}
                >
                    {m}
                </span>
            ))}
        </div>
    );
}
