"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useMockData } from "@/lib/MockDataProvider";

type TabKey = "info" | "progress" | "records" | "members";

const statusColors: Record<string, string> = {
    "已通过": "bg-green-50 text-green-600",
    "待审核": "bg-yellow-50 text-yellow-600",
    "已退回": "bg-red-50 text-red-600",
};

export default function ProjectDetailPage() {
    const params = useParams();
    const router = useRouter();
    const { data, getCumulativeProgress } = useMockData();
    const [activeTab, setActiveTab] = useState<TabKey>("info");

    const projectId = Number(params.id);
    const project = data.projects.find((p) => p.id === projectId);

    if (!project) {
        return (
            <div className="flex flex-col items-center justify-center py-20 text-gray-400">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                    <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <p className="mt-3 text-sm">项目不存在</p>
                <button
                    onClick={() => router.push("/pc/projects")}
                    className="mt-3 text-sm text-eng-blue hover:underline"
                >
                    返回项目列表
                </button>
            </div>
        );
    }

    const leader = data.users.find((u) => u.id === project.teamLeaderId);
    const dept = data.departments.find((d) => d.id === project.departmentId);
    const creator = data.users.find((u) => u.id === project.creatorId);

    const members = project.memberIds
        .map((id) => data.workers.find((w) => w.id === id))
        .filter((w): w is NonNullable<typeof w> => w !== undefined);

    const projectRecords = data.dailyRecords
        .filter((r) => r.projectId === projectId)
        .sort((a, b) => b.date.localeCompare(a.date));

    const overallProgress = project.workItems.reduce((sum, wi) => {
        const cumulative = getCumulativeProgress(projectId, wi.id);
        const pct = Math.min(cumulative / wi.targetQuantity, 1);
        return sum + pct * wi.weight;
    }, 0);

    const tabs: { key: TabKey; label: string }[] = [
        { key: "info", label: "基本信息" },
        { key: "progress", label: "工程量进度" },
        { key: "records", label: "近期日报" },
        { key: "members", label: "项目成员" },
    ];

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-3">
                <button
                    onClick={() => router.push("/pc/projects")}
                    className="text-gray-400 hover:text-gray-600"
                >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                        <path d="M15 19l-7-7 7-7" />
                    </svg>
                </button>
                <div>
                    <h1 className="text-xl font-bold text-gray-800">
                        {project.name}
                    </h1>
                    <div className="flex items-center gap-2 mt-1">
                        <span className="text-xs px-2 py-0.5 bg-blue-50 text-eng-blue rounded-full">
                            {project.status}
                        </span>
                        {project.isSubcontractor && (
                            <span className="text-xs px-2 py-0.5 bg-purple-50 text-purple-600 rounded-full">
                                分包项目
                            </span>
                        )}
                        <span className="text-xs text-gray-400">
                            总进度 {(overallProgress * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="border-b border-gray-200">
                <div className="flex gap-6">
                    {tabs.map((tab) => (
                        <button
                            key={tab.key}
                            onClick={() => setActiveTab(tab.key)}
                            className={`pb-2.5 text-sm font-medium border-b-2 transition-colors ${
                                activeTab === tab.key
                                    ? "border-eng-blue text-eng-blue"
                                    : "border-transparent text-gray-500 hover:text-gray-700"
                            }`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Tab Content */}
            {activeTab === "info" && (
                <InfoTab
                    project={project}
                    leader={leader}
                    dept={dept}
                    creator={creator}
                />
            )}
            {activeTab === "progress" && (
                <ProgressTab
                    project={project}
                    projectId={projectId}
                    getCumulativeProgress={getCumulativeProgress}
                />
            )}
            {activeTab === "records" && (
                <RecordsTab records={projectRecords} />
            )}
            {activeTab === "members" && (
                <MembersTab members={members} />
            )}
        </div>
    );
}

function InfoTab({
    project,
    leader,
    dept,
    creator,
}: {
    project: {
        startDate: string;
        endDate: string;
        remarks: string;
        createdAt: string;
        constructionPlanUrls: string[];
        isSubcontractor: boolean;
    };
    leader: { name: string } | undefined;
    dept: { name: string } | undefined;
    creator: { name: string } | undefined;
}) {
    const fields = [
        { label: "所属部门", value: dept?.name || "-" },
        { label: "班组长", value: leader?.name || "-" },
        { label: "创建人", value: creator?.name || "-" },
        { label: "开工日期", value: project.startDate },
        { label: "竣工日期", value: project.endDate },
        { label: "创建时间", value: project.createdAt.split("T")[0] },
        { label: "分包项目", value: project.isSubcontractor ? "是" : "否" },
        { label: "施工方案", value: project.constructionPlanUrls.length > 0 ? `${project.constructionPlanUrls.length} 份` : "未上传" },
    ];

    return (
        <div className="bg-white rounded-lg border border-gray-200">
            <div className="px-5 py-3 border-b border-gray-100">
                <h2 className="text-sm font-semibold text-gray-700">
                    项目基本信息
                </h2>
            </div>
            <div className="p-5">
                <div className="grid grid-cols-2 gap-x-12 gap-y-4">
                    {fields.map((f) => (
                        <div key={f.label} className="flex items-start gap-3">
                            <span className="text-sm text-gray-400 w-20 shrink-0">
                                {f.label}
                            </span>
                            <span className="text-sm text-gray-800">
                                {f.value}
                            </span>
                        </div>
                    ))}
                </div>
                {project.remarks && (
                    <div className="mt-4 pt-4 border-t border-gray-100">
                        <span className="text-sm text-gray-400">备注：</span>
                        <span className="text-sm text-gray-600 ml-2">
                            {project.remarks}
                        </span>
                    </div>
                )}
            </div>
        </div>
    );
}

function ProgressTab({
    project,
    projectId,
    getCumulativeProgress,
}: {
    project: { workItems: Array<{ id: number; name: string; targetQuantity: number; unit: string; weight: number }> };
    projectId: number;
    getCumulativeProgress: (projectId: number, workItemId: number) => number;
}) {
    return (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-gray-100 bg-gray-50">
                        <th className="text-left px-4 py-3 text-gray-500 font-medium">工序名称</th>
                        <th className="text-right px-4 py-3 text-gray-500 font-medium">目标量</th>
                        <th className="text-left px-4 py-3 text-gray-500 font-medium">单位</th>
                        <th className="text-right px-4 py-3 text-gray-500 font-medium">权重</th>
                        <th className="text-right px-4 py-3 text-gray-500 font-medium">累计完成</th>
                        <th className="text-left px-4 py-3 text-gray-500 font-medium w-48">进度</th>
                    </tr>
                </thead>
                <tbody>
                    {project.workItems.map((wi) => {
                        const cumulative = getCumulativeProgress(projectId, wi.id);
                        const pct = Math.min((cumulative / wi.targetQuantity) * 100, 100);
                        return (
                            <tr key={wi.id} className="border-b border-gray-50">
                                <td className="px-4 py-3 text-gray-800 font-medium">{wi.name}</td>
                                <td className="px-4 py-3 text-gray-600 text-right">{wi.targetQuantity.toLocaleString()}</td>
                                <td className="px-4 py-3 text-gray-500">{wi.unit}</td>
                                <td className="px-4 py-3 text-gray-500 text-right">{(wi.weight * 100).toFixed(0)}%</td>
                                <td className="px-4 py-3 text-gray-800 text-right">{cumulative.toLocaleString()}</td>
                                <td className="px-4 py-3">
                                    <div className="flex items-center gap-2">
                                        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-eng-blue rounded-full"
                                                style={{ width: `${pct}%` }}
                                            />
                                        </div>
                                        <span className="text-xs text-gray-500 w-12 text-right">
                                            {pct.toFixed(1)}%
                                        </span>
                                    </div>
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

function RecordsTab({ records }: { records: Array<{ id: number; date: string; moduleType: string; status: string; teamLeaderId: number }> }) {
    if (records.length === 0) {
        return (
            <div className="bg-white rounded-lg border border-gray-200 py-12 text-center text-gray-400 text-sm">
                暂无日报记录
            </div>
        );
    }

    const moduleLabels: Record<string, string> = {
        "考勤": "安全签到",
        "进度": "施工进度",
        "材料": "材料设备",
        "施工记录": "施工记录",
    };

    return (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-gray-100 bg-gray-50">
                        <th className="text-left px-4 py-3 text-gray-500 font-medium">日期</th>
                        <th className="text-left px-4 py-3 text-gray-500 font-medium">模块</th>
                        <th className="text-left px-4 py-3 text-gray-500 font-medium">状态</th>
                        <th className="text-left px-4 py-3 text-gray-500 font-medium">班组长</th>
                    </tr>
                </thead>
                <tbody>
                    {records.map((r) => (
                        <tr key={r.id} className="border-b border-gray-50 hover:bg-gray-50">
                            <td className="px-4 py-3 text-gray-800">{r.date}</td>
                            <td className="px-4 py-3 text-gray-600">{moduleLabels[r.moduleType] || r.moduleType}</td>
                            <td className="px-4 py-3">
                                <span className={`text-xs px-2 py-0.5 rounded-full ${statusColors[r.status] || "bg-gray-50 text-gray-500"}`}>
                                    {r.status}
                                </span>
                            </td>
                            <td className="px-4 py-3 text-gray-500">-</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function MembersTab({ members }: { members: Array<{ id: number; name: string; position: string; certifications: Array<{ type: string; level: string; expireDate: string }> }> }) {
    if (members.length === 0) {
        return (
            <div className="bg-white rounded-lg border border-gray-200 py-12 text-center text-gray-400 text-sm">
                暂无项目成员
            </div>
        );
    }

    return (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-gray-100 bg-gray-50">
                        <th className="text-left px-4 py-3 text-gray-500 font-medium">姓名</th>
                        <th className="text-left px-4 py-3 text-gray-500 font-medium">岗位</th>
                        <th className="text-left px-4 py-3 text-gray-500 font-medium">资质证书</th>
                        <th className="text-left px-4 py-3 text-gray-500 font-medium">有效期</th>
                    </tr>
                </thead>
                <tbody>
                    {members.map((m) => (
                        <tr key={m.id} className="border-b border-gray-50">
                            <td className="px-4 py-3 text-gray-800 font-medium">{m.name}</td>
                            <td className="px-4 py-3 text-gray-600">{m.position}</td>
                            <td className="px-4 py-3">
                                {m.certifications.length > 0 ? (
                                    <div className="space-y-0.5">
                                        {m.certifications.map((c, i) => (
                                            <div key={i} className="text-xs text-gray-500">
                                                {c.type} {c.level}
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <span className="text-xs text-gray-400">无</span>
                                )}
                            </td>
                            <td className="px-4 py-3">
                                {m.certifications.length > 0 ? (
                                    <div className="space-y-0.5">
                                        {m.certifications.map((c, i) => (
                                            <div key={i} className="text-xs text-gray-500">
                                                {c.expireDate}
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <span className="text-xs text-gray-400">-</span>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
