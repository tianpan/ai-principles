"use client";

import { useMemo } from "react";
import { useMockData } from "@/lib/MockDataProvider";
import { AiBriefingCard } from "@/components/ai/AiBriefingCard";
import { generateBriefing } from "@/lib/ai/briefing-generator";

export default function PcDashboardPage() {
    const {
        currentUser,
        data,
        getCumulativeProgress,
        getPendingRecords,
        getOverdueRecords,
        getAiBriefingByProjectAndDate,
        addAiBriefing,
    } = useMockData();

    const projects = data.projects;
    const pendingCount = getPendingRecords().length;
    const overdueCount = getOverdueRecords().length;

    // --- 今日简报 ---
    const today = new Date().toISOString().split("T")[0];
    const firstProject = projects[0];

    const briefing = useMemo(() => {
        if (!firstProject) return null;
        const existing = getAiBriefingByProjectAndDate(firstProject.id, today);
        if (existing) return existing;

        // 生成新简报
        const todayRecords = data.dailyRecords.filter(
            (r) => r.projectId === firstProject.id && r.date === today
        );
        const progressItems = firstProject.workItems.map((wi) => {
            const cumulative = getCumulativeProgress(firstProject.id, wi.id);
            const todayQty = todayRecords
                .filter((r) => r.moduleType === "进度")
                .reduce((sum, r) => {
                    const prog = r.progress as Array<Record<string, unknown>> | undefined;
                    return sum + (prog
                        ? prog
                            .filter((p) => p.workItemId === wi.id)
                            .reduce((s, p) => s + ((p.quantity as number) ?? 0), 0)
                        : 0);
                }, 0);
            return {
                name: wi.name,
                todayQuantity: todayQty,
                cumulativeQuantity: cumulative,
                targetQuantity: wi.targetQuantity,
                unit: wi.unit,
            };
        });

        const materialLines = todayRecords
            .filter((r) => r.moduleType === "材料")
            .map((r) => {
                const mats = r.materials as Array<Record<string, unknown>> | undefined;
                return mats
                    ? mats.map((m) => `${m.recordType === "退料" ? "退" : "领"}：${m.name} ${m.quantity}${m.unit}`).join("；")
                    : "";
            })
            .filter(Boolean);

        const attendanceRecords = todayRecords.filter((r) => r.moduleType === "考勤");
        const attendeeCount = attendanceRecords.reduce((sum, r) => {
            const att = r.attendance as Record<string, unknown> | undefined;
            return sum + ((att?.attendeeCount as number) ?? 0);
        }, 0);

        const risks: string[] = [];
        for (const item of progressItems) {
            if (item.targetQuantity > 0) {
                const pct = item.cumulativeQuantity / item.targetQuantity;
                const daysTotal = Math.ceil(
                    (new Date(firstProject.endDate).getTime() -
                        new Date(firstProject.startDate).getTime()) /
                        (1000 * 60 * 60 * 24)
                );
                const daysElapsed = Math.ceil(
                    (Date.now() - new Date(firstProject.startDate).getTime()) /
                        (1000 * 60 * 60 * 24)
                );
                const expectedPct = daysTotal > 0 ? daysElapsed / daysTotal : 0;
                if (pct < expectedPct - 0.1) {
                    risks.push(`${item.name} 进度 ${Math.round(pct * 100)}%，低于计划 ${Math.round(expectedPct * 100)}%，存在滞后风险`);
                }
            }
        }

        const output = generateBriefing({
            projectId: firstProject.id,
            projectName: firstProject.name,
            date: today,
            progressItems,
            materialSummary: materialLines.length > 0 ? materialLines.join("\n") : "暂无材料记录",
            attendanceSummary: attendeeCount > 0 ? `出勤人数：${attendeeCount}人` : "暂无考勤记录",
            risks,
        });

        return addAiBriefing({
            projectId: output.projectId,
            date: output.date,
            content: output.content,
            risks: output.risks,
            createdAt: new Date().toISOString(),
        });
    }, [firstProject, today, data.dailyRecords, getCumulativeProgress, getAiBriefingByProjectAndDate, addAiBriefing]);

    if (!currentUser) return null;

    return (
        <div className="space-y-6">
            <h1 className="text-xl font-bold text-gray-800">工作台</h1>

            {/* AI 每日简报 */}
            {briefing && (
                <AiBriefingCard
                    content={briefing.content}
                    risks={briefing.risks}
                    date={briefing.date}
                    projectName={firstProject?.name ?? ""}
                />
            )}

            {/* Alert Cards */}
            <div className="grid grid-cols-4 gap-4">
                <StatCard
                    label="进行中项目"
                    value={projects.filter((p) => p.status === "进行中").length}
                    color="text-eng-blue"
                    bg="bg-blue-50"
                />
                <StatCard
                    label="待审核"
                    value={pendingCount}
                    color="text-eng-orange"
                    bg="bg-orange-50"
                />
                <StatCard
                    label="超时预警"
                    value={overdueCount}
                    color="text-eng-red"
                    bg="bg-red-50"
                />
                <StatCard
                    label="今日已通过"
                    value={
                        data.dailyRecords.filter(
                            (r) =>
                                r.status === "已通过" &&
                                r.date ===
                                    new Date().toISOString().split("T")[0]
                        ).length
                    }
                    color="text-eng-green"
                    bg="bg-green-50"
                />
            </div>

            {/* Project Progress Grid */}
            <div>
                <h2 className="text-base font-medium text-gray-700 mb-3">
                    项目进度概览
                </h2>
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
                                <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                    工期
                                </th>
                                <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                    综合进度
                                </th>
                                <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                    状态
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {projects.map((project) => {
                                const leader = data.users.find(
                                    (u) => u.id === project.teamLeaderId
                                );
                                const overallProgress =
                                    project.workItems.reduce((sum, wi) => {
                                        const cumulative =
                                            getCumulativeProgress(
                                                project.id,
                                                wi.id
                                            );
                                        const pct = Math.min(
                                            cumulative / wi.targetQuantity,
                                            1
                                        );
                                        return sum + pct * wi.weight;
                                    }, 0);
                                const daysLeft = Math.ceil(
                                    (new Date(project.endDate).getTime() -
                                        Date.now()) /
                                        (1000 * 60 * 60 * 24)
                                );

                                return (
                                    <tr
                                        key={project.id}
                                        className="border-b border-gray-50 hover:bg-gray-50"
                                    >
                                        <td className="px-4 py-3 text-gray-800">
                                            {project.name}
                                        </td>
                                        <td className="px-4 py-3 text-gray-600">
                                            {leader?.name || "-"}
                                        </td>
                                        <td className="px-4 py-3 text-gray-600">
                                            <span className="text-xs">
                                                {project.startDate} ~{" "}
                                                {project.endDate}
                                            </span>
                                            <br />
                                            <span
                                                className={`text-xs ${
                                                    daysLeft < 0
                                                        ? "text-eng-red"
                                                        : daysLeft < 15
                                                        ? "text-eng-orange"
                                                        : "text-eng-green"
                                                }`}
                                            >
                                                {daysLeft > 0
                                                    ? `剩余 ${daysLeft} 天`
                                                    : daysLeft === 0
                                                    ? "今日截止"
                                                    : `超期 ${Math.abs(daysLeft)} 天`}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <div className="flex items-center gap-2">
                                                <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden max-w-[120px]">
                                                    <div
                                                        className="h-full bg-eng-blue rounded-full"
                                                        style={{
                                                            width: `${Math.min(overallProgress * 100, 100)}%`,
                                                        }}
                                                    />
                                                </div>
                                                <span className="text-xs text-gray-500 w-12 text-right">
                                                    {(
                                                        overallProgress * 100
                                                    ).toFixed(1)}
                                                    %
                                                </span>
                                            </div>
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className="text-xs px-2 py-0.5 bg-blue-50 text-eng-blue rounded-full">
                                                {project.status}
                                            </span>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Safety Summary */}
            <div>
                <h2 className="text-base font-medium text-gray-700 mb-3">
                    安全交底统计
                </h2>
                <div className="grid grid-cols-3 gap-4">
                    {data.templates.map((t) => {
                        const count = data.dailyRecords.filter(
                            (r) =>
                                r.moduleType === "考勤" &&
                                r.status === "已通过" &&
                                r.attendance &&
                                (r.attendance as Record<string, unknown>)
                                    .safetyBriefingType === t.category
                        ).length;
                        return (
                            <div
                                key={t.id}
                                className="bg-white rounded-lg border border-gray-200 p-4"
                            >
                                <p className="text-sm text-gray-700 font-medium">
                                    {t.name}
                                </p>
                                <p className="text-2xl font-bold text-eng-blue mt-1">
                                    {count}
                                </p>
                                <p className="text-xs text-gray-400">
                                    次安全交底
                                </p>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}

function StatCard({
    label,
    value,
    color,
    bg,
}: {
    label: string;
    value: number;
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
