"use client";

import { useState } from "react";
import { useMockData } from "@/lib/MockDataProvider";

export default function ReviewPage() {
    const { currentUser, data, updateDailyRecord, getProjectById, getUserById } =
        useMockData();
    const [expandedId, setExpandedId] = useState<number | null>(null);
    const [returnReason, setReturnReason] = useState("");
    const [activeReasonId, setActiveReasonId] = useState<number | null>(null);

    if (!currentUser) return null;

    const isTeamLeader = currentUser.role === "班组长";
    const isPM = currentUser.role === "项目经理";

    // Team leaders see their own pending; PM sees all
    const pendingRecords = data.dailyRecords.filter(
        (r) =>
            r.status === "待审核" &&
            (isPM || (isTeamLeader && r.teamLeaderId === currentUser.id))
    );

    const overdueRecords = data.dailyRecords.filter((r) => {
        if (r.status !== "待审核") return false;
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
        return r.date < sevenDaysAgo.toISOString().split("T")[0];
    });

    const handleApprove = (id: number) => {
        updateDailyRecord(id, {
            status: "已通过",
            reviewComment: "审核通过",
            reviewedAt: new Date().toISOString(),
        });
    };

    const handleReturn = (id: number) => {
        if (!returnReason.trim()) return;
        updateDailyRecord(id, {
            status: "已退回",
            reviewComment: returnReason,
            reviewedAt: new Date().toISOString(),
        });
        setReturnReason("");
        setActiveReasonId(null);
    };

    return (
        <div className="px-4 py-4 space-y-4">
            <h1 className="text-app-title font-bold text-eng-gray-900">
                审核管理
            </h1>

            {/* Overdue Warning */}
            {overdueRecords.length > 0 && (
                <div className="bg-red-50 border border-eng-red/20 rounded-card p-3">
                    <div className="flex items-center gap-2 mb-1">
                        <svg
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth={2}
                            className="text-eng-red"
                        >
                            <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
                        </svg>
                        <span className="text-sm font-medium text-eng-red">
                            超时提醒
                        </span>
                    </div>
                    <p className="text-xs text-eng-red/80">
                        {overdueRecords.length} 条记录已超过7天未审核
                    </p>
                </div>
            )}

            {/* Pending Count */}
            <div className="flex items-center justify-between">
                <span className="text-sm text-eng-gray-500">
                    待审核 {pendingRecords.length} 条
                </span>
            </div>

            {/* Records */}
            <div className="space-y-3">
                {pendingRecords.map((record) => {
                    const project = getProjectById(record.projectId);
                    const leader = getUserById(record.teamLeaderId);
                    const isExpanded = expandedId === record.id;

                    return (
                        <div
                            key={record.id}
                            className="bg-white rounded-card border border-eng-gray-100 overflow-hidden"
                        >
                            {/* Header */}
                            <button
                                onClick={() =>
                                    setExpandedId(isExpanded ? null : record.id)
                                }
                                className="w-full p-3 flex items-center justify-between text-left"
                            >
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                        <span className="text-sm font-medium text-eng-gray-800">
                                            {record.moduleType}
                                        </span>
                                        <span className="text-xs text-eng-gray-400">
                                            {record.date}
                                        </span>
                                    </div>
                                    <p className="text-xs text-eng-gray-400 truncate mt-0.5">
                                        {project?.name} · {leader?.name}
                                    </p>
                                </div>
                                <svg
                                    width="16"
                                    height="16"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth={1.5}
                                    className={`text-eng-gray-400 transition-transform ${
                                        isExpanded ? "rotate-180" : ""
                                    }`}
                                >
                                    <path d="M19 9l-7 7-7-7" />
                                </svg>
                            </button>

                            {/* Expanded Detail */}
                            {isExpanded && (
                                <div className="px-3 pb-3 border-t border-eng-gray-50 pt-2 space-y-3">
                                    <RecordDetail record={record} />

                                    {/* Action Buttons (PM only) */}
                                    {isPM && (
                                        <div className="space-y-2">
                                            <div className="flex gap-2">
                                                <button
                                                    onClick={() =>
                                                        handleApprove(record.id)
                                                    }
                                                    className="flex-1 h-9 bg-eng-green text-white rounded-btn text-sm font-medium"
                                                >
                                                    通过
                                                </button>
                                                <button
                                                    onClick={() =>
                                                        setActiveReasonId(
                                                            activeReasonId === record.id
                                                                ? null
                                                                : record.id
                                                        )
                                                    }
                                                    className="flex-1 h-9 bg-eng-red text-white rounded-btn text-sm font-medium"
                                                >
                                                    退回
                                                </button>
                                            </div>
                                            {activeReasonId === record.id && (
                                                <div className="space-y-2">
                                                    <textarea
                                                        value={returnReason}
                                                        onChange={(e) =>
                                                            setReturnReason(
                                                                e.target.value
                                                            )
                                                        }
                                                        placeholder="请输入退回原因"
                                                        className="w-full h-16 px-3 py-2 border border-eng-gray-200 rounded-btn text-sm resize-none focus:outline-none focus:border-eng-red"
                                                    />
                                                    <button
                                                        onClick={() =>
                                                            handleReturn(record.id)
                                                        }
                                                        disabled={
                                                            !returnReason.trim()
                                                        }
                                                        className={`w-full h-9 rounded-btn text-sm font-medium ${
                                                            returnReason.trim()
                                                                ? "bg-eng-red text-white"
                                                                : "bg-eng-gray-200 text-eng-gray-400"
                                                        }`}
                                                    >
                                                        确认退回
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    );
                })}

                {pendingRecords.length === 0 && (
                    <p className="text-sm text-eng-gray-400 text-center py-8">
                        暂无待审核记录
                    </p>
                )}
            </div>
        </div>
    );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function RecordDetail({ record }: { record: any }) {
    const r = record as {
        moduleType: string;
        attendance?: Record<string, unknown>;
        progress?: Array<Record<string, unknown>>;
        materials?: Array<Record<string, unknown>>;
        siteRecord?: Record<string, unknown>;
    };

    if (r.moduleType === "考勤" && r.attendance) {
        const a = r.attendance;
        return (
            <div className="text-xs text-eng-gray-600 space-y-1">
                <p>GPS: {(a.gpsLocation as string) || "-"}</p>
                <p>
                    安全交底: {(a.safetyBriefingType as string) || "-"}
                </p>
                <p>
                    参会:{" "}
                    {((a.attendeeIds as number[]) || []).length} 人
                </p>
            </div>
        );
    }

    if (r.moduleType === "进度" && r.progress) {
        return (
            <div className="text-xs text-eng-gray-600 space-y-1">
                {(r.progress as Array<{ workItemId: number; completedQuantity: number }>).map(
                    (p, i) => (
                        <p key={i}>
                            工序#{p.workItemId}: {p.completedQuantity}
                        </p>
                    )
                )}
            </div>
        );
    }

    if (r.moduleType === "材料" && r.materials) {
        return (
            <div className="text-xs text-eng-gray-600 space-y-1">
                {(r.materials as Array<{ materialName: string; quantity: number; recordType: string }>).map(
                    (m, i) => (
                        <p key={i}>
                            [{m.recordType}] {m.materialName} x{m.quantity}
                        </p>
                    )
                )}
            </div>
        );
    }

    if (r.moduleType === "施工记录" && r.siteRecord) {
        const s = r.siteRecord as { formData?: Record<string, unknown> };
        return (
            <div className="text-xs text-eng-gray-600 space-y-1">
                {s.formData &&
                    Object.entries(s.formData).map(([k, v]) => (
                        <p key={k}>
                            {k}: {String(v)}
                        </p>
                    ))}
            </div>
        );
    }

    return <p className="text-xs text-eng-gray-400">无详细信息</p>;
}
