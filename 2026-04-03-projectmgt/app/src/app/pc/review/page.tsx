"use client";

import { useState, useMemo, useCallback } from "react";
import { useMockData } from "@/lib/MockDataProvider";
import { AiReviewSummary } from "@/components/ai/AiReviewSummary";
import { AiReviewDetail } from "@/components/ai/AiReviewDetail";
import { preReview, type PreReviewResult } from "@/lib/ai/smart-review";
import type { AuditRecord, AuditContext } from "@/lib/ai/rule-engine";

export default function PcReviewPage() {
    const {
        currentUser,
        data,
        updateDailyRecord,
        getProjectById,
        getUserById,
    } = useMockData();
    const [filterStatus, setFilterStatus] = useState("待审核");
    const [filterProject, setFilterProject] = useState("全部");
    const [selectedRecord, setSelectedRecord] = useState<number | null>(null);
    const [returnReason, setReturnReason] = useState("");
    const [detailRecord, setDetailRecord] = useState<{
        recordId: number;
        moduleType: string;
        date: string;
        results: import("@/lib/ai/rule-engine").AuditResult[];
    } | null>(null);

    const records = useMemo(() => {
        return data.dailyRecords.filter((r) => {
            if (filterStatus !== "全部" && r.status !== filterStatus)
                return false;
            if (filterProject !== "全部" && r.projectId !== Number(filterProject))
                return false;
            return true;
        });
    }, [data.dailyRecords, filterStatus, filterProject]);

    const handleApprove = useCallback((id: number) => {
        updateDailyRecord(id, {
            status: "已通过",
            reviewComment: "审核通过",
            reviewedAt: new Date().toISOString(),
        });
    }, [updateDailyRecord]);

    const handleReturn = (id: number) => {
        if (!returnReason.trim()) return;
        updateDailyRecord(id, {
            status: "已退回",
            reviewComment: returnReason,
            reviewedAt: new Date().toISOString(),
        });
        setReturnReason("");
        setSelectedRecord(null);
    };

    const handleApproveAll = (teamLeaderId: number) => {
        const pending = records.filter(
            (r) => r.teamLeaderId === teamLeaderId && r.status === "待审核"
        );
        pending.forEach((r) => handleApprove(r.id));
    };

    // --- AI 预审 ---
    const pendingRecords = useMemo(
        () => records.filter((r) => r.status === "待审核"),
        [records]
    );

    const auditContext: AuditContext = useMemo(
        () => {
            const projectId = filterProject === "全部" ? undefined : Number(filterProject);
            const project = projectId
                ? data.projects.find((p) => p.id === projectId)
                : data.projects[0];
            const today = new Date().toISOString().slice(0, 10);
            const todayRecords = data.dailyRecords.filter(
                (r) => r.date === today && (!projectId || r.projectId === projectId)
            );
            const todayAttRecord = todayRecords.find((r) => r.moduleType === "考勤");
            const att = todayAttRecord?.attendance as Record<string, unknown> | undefined;

            return {
                history: data.dailyRecords
                    .filter((r) => !projectId || r.projectId === projectId)
                    .map((r) => ({
                        date: r.date,
                        moduleType: r.moduleType,
                        progress: r.progress,
                        attendance: r.attendance,
                        materials: r.materials,
                    })),
                project: project
                    ? {
                        id: project.id,
                        name: project.name,
                        startDate: project.startDate,
                        endDate: project.endDate,
                        workItems: project.workItems.map((wi) => ({
                            name: wi.name,
                            targetQuantity: wi.targetQuantity,
                            unit: wi.unit,
                            weight: wi.weight,
                        })),
                    }
                    : { id: 0, name: "", startDate: "", endDate: "", workItems: [] },
                todayAttendance: att
                    ? { attendeeCount: (att.attendeeCount as number) ?? 0 }
                    : null,
                workers: data.workers.map((w) => ({
                    name: w.name,
                    position: w.position ?? "",
                    certifications: (w.certifications ?? []).map((c) => ({
                        type: c.type,
                        level: c.level ?? "",
                        expireDate: c.expireDate ?? "",
                    })),
                })),
            };
        },
        [data.dailyRecords, data.projects, data.workers, filterProject]
    );

    const auditRecords: AuditRecord[] = useMemo(
        () =>
            pendingRecords.map((r) => ({
                id: r.id,
                projectId: r.projectId,
                date: r.date,
                moduleType: r.moduleType,
                status: r.status,
                progress: r.progress,
                attendance: r.attendance,
                materials: r.materials,
                siteRecord: r.siteRecord,
            })),
        [pendingRecords]
    );

    const preReviewResult: PreReviewResult = useMemo(
        () => preReview(auditRecords, auditContext),
        [auditRecords, auditContext]
    );

    const handleBatchApprove = useCallback(() => {
        for (const record of preReviewResult.normal) {
            handleApprove(record.id);
        }
    }, [preReviewResult.normal, handleApprove]);

    const handleDetailApprove = useCallback(() => {
        if (detailRecord) {
            handleApprove(detailRecord.recordId);
            setDetailRecord(null);
        }
    }, [detailRecord, handleApprove]);

    const handleDetailReject = useCallback(() => {
        if (detailRecord) {
            setSelectedRecord(detailRecord.recordId);
            setDetailRecord(null);
        }
    }, [detailRecord]);

    // Group by team leader
    const grouped = useMemo(() => {
        const map = new Map<number, typeof records>();
        for (const r of records) {
            const existing = map.get(r.teamLeaderId) || [];
            existing.push(r);
            map.set(r.teamLeaderId, existing);
        }
        return map;
    }, [records]);

    if (!currentUser) return null;

    return (
        <div className="space-y-4">
            <h1 className="text-xl font-bold text-gray-800">审核工作台</h1>

            {/* AI 预审摘要 */}
            {pendingRecords.length > 0 && (
                <AiReviewSummary
                    result={preReviewResult}
                    onBatchApprove={handleBatchApprove}
                />
            )}

            {/* AI 异常详情 */}
            {detailRecord && (
                <div className="mb-2">
                    <AiReviewDetail
                        recordId={detailRecord.recordId}
                        moduleType={detailRecord.moduleType}
                        date={detailRecord.date}
                        results={detailRecord.results}
                        onApprove={handleDetailApprove}
                        onReject={handleDetailReject}
                    />
                </div>
            )}

            {/* Filters */}
            <div className="flex gap-3">
                <select
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                    className="h-9 px-3 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                >
                    <option value="全部">全部状态</option>
                    <option value="待审核">待审核</option>
                    <option value="已通过">已通过</option>
                    <option value="已退回">已退回</option>
                </select>
                <select
                    value={filterProject}
                    onChange={(e) => setFilterProject(e.target.value)}
                    className="h-9 px-3 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                >
                    <option value="全部">全部项目</option>
                    {data.projects.map((p) => (
                        <option key={p.id} value={p.id}>
                            {p.name}
                        </option>
                    ))}
                </select>
                <span className="text-sm text-gray-400 self-center ml-auto">
                    共 {records.length} 条
                </span>
            </div>

            {/* By Team Leader */}
            {Array.from(grouped.entries()).map(([leaderId, leaderRecords]) => {
                const leader = getUserById(leaderId);
                const pendingCount = leaderRecords.filter(
                    (r) => r.status === "待审核"
                ).length;

                return (
                    <div
                        key={leaderId}
                        className="bg-white rounded-lg border border-gray-200 overflow-hidden"
                    >
                        <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-b border-gray-100">
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-medium text-gray-700">
                                    {leader?.name}
                                </span>
                                <span className="text-xs text-gray-400">
                                    {pendingCount} 条待审核
                                </span>
                            </div>
                            {pendingCount > 0 && (
                                <button
                                    onClick={() => handleApproveAll(leaderId)}
                                    className="text-xs text-eng-blue hover:text-blue-600"
                                >
                                    全部通过
                                </button>
                            )}
                        </div>

                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-gray-100 bg-gray-50/50">
                                    <th className="text-left px-4 py-2 text-gray-500 font-medium text-xs">
                                        日期
                                    </th>
                                    <th className="text-left px-4 py-2 text-gray-500 font-medium text-xs">
                                        项目
                                    </th>
                                    <th className="text-left px-4 py-2 text-gray-500 font-medium text-xs">
                                        模块
                                    </th>
                                    <th className="text-left px-4 py-2 text-gray-500 font-medium text-xs">
                                        状态
                                    </th>
                                    <th className="text-right px-4 py-2 text-gray-500 font-medium text-xs">
                                        操作
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                {leaderRecords.map((record) => {
                                    const project = getProjectById(
                                        record.projectId
                                    );
                                    const isPending =
                                        record.status === "待审核";
                                    const isSelected =
                                        selectedRecord === record.id;

                                    return (
                                        <tr
                                            key={record.id}
                                            className="border-b border-gray-50 hover:bg-gray-50"
                                        >
                                            <td className="px-4 py-2.5 text-gray-600 text-xs">
                                                {record.date}
                                            </td>
                                            <td className="px-4 py-2.5 text-gray-700 text-xs max-w-[200px] truncate">
                                                {project?.name}
                                            </td>
                                            <td className="px-4 py-2.5">
                                                <span className="text-xs px-1.5 py-0.5 bg-gray-100 rounded">
                                                    {record.moduleType}
                                                </span>
                                            </td>
                                            <td className="px-4 py-2.5">
                                                <StatusBadge
                                                    status={record.status}
                                                />
                                            </td>
                                            <td className="px-4 py-2.5 text-right">
                                                {isPending ? (
                                                    <div className="flex items-center justify-end gap-1">
                                                        <button
                                                            onClick={() => {
                                                                const susItem = preReviewResult.suspicious.find((s) => s.record.id === record.id);
                                                                const critItem = preReviewResult.critical.find((c) => c.record.id === record.id);
                                                                const results = susItem?.results || critItem?.results || [];
                                                                setDetailRecord({
                                                                    recordId: record.id,
                                                                    moduleType: record.moduleType,
                                                                    date: record.date,
                                                                    results,
                                                                });
                                                            }}
                                                            className="text-xs text-eng-blue hover:underline"
                                                        >
                                                            AI详情
                                                        </button>
                                                        <button
                                                            onClick={() =>
                                                                handleApprove(
                                                                    record.id
                                                                )
                                                            }
                                                            className="text-xs text-eng-green hover:underline"
                                                        >
                                                            通过
                                                        </button>
                                                        <button
                                                            onClick={() =>
                                                                setSelectedRecord(
                                                                    isSelected
                                                                        ? null
                                                                        : record.id
                                                                )
                                                            }
                                                            className="text-xs text-eng-red hover:underline"
                                                        >
                                                            退回
                                                        </button>
                                                    </div>
                                                ) : (
                                                    <span className="text-xs text-gray-400">
                                                        {record.reviewComment}
                                                    </span>
                                                )}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                );
            })}

            {/* Return Reason Modal */}
            {selectedRecord && (
                <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
                    <div className="bg-white rounded-lg p-6 w-96 space-y-4">
                        <h3 className="text-base font-medium text-gray-800">
                            退回原因
                        </h3>
                        <textarea
                            value={returnReason}
                            onChange={(e) => setReturnReason(e.target.value)}
                            placeholder="请输入退回原因"
                            className="w-full h-24 px-3 py-2 border border-gray-200 rounded text-sm resize-none focus:outline-none focus:border-eng-red"
                        />
                        <div className="flex justify-end gap-2">
                            <button
                                onClick={() => {
                                    setSelectedRecord(null);
                                    setReturnReason("");
                                }}
                                className="h-9 px-4 border border-gray-200 rounded text-sm text-gray-600"
                            >
                                取消
                            </button>
                            <button
                                onClick={() => handleReturn(selectedRecord)}
                                disabled={!returnReason.trim()}
                                className={`h-9 px-4 rounded text-sm font-medium ${
                                    returnReason.trim()
                                        ? "bg-eng-red text-white"
                                        : "bg-gray-200 text-gray-400"
                                }`}
                            >
                                确认退回
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

function StatusBadge({ status }: { status: string }) {
    const styles: Record<string, string> = {
        已通过: "bg-green-50 text-eng-green",
        待审核: "bg-orange-50 text-eng-orange",
        已退回: "bg-red-50 text-eng-red",
    };
    return (
        <span
            className={`text-xs px-2 py-0.5 rounded-full ${
                styles[status] || "bg-gray-50 text-gray-500"
            }`}
        >
            {status}
        </span>
    );
}
