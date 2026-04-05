"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMockData } from "@/lib/MockDataProvider";
import Link from "next/link";

export default function PcProjectsPage() {
    const { currentUser, data, getCumulativeProgress } = useMockData();
    const router = useRouter();
    const [search, setSearch] = useState("");

    if (!currentUser) return null;

    const filtered = data.projects.filter((p) =>
        p.name.includes(search)
    );

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <h1 className="text-xl font-bold text-gray-800">
                    项目管理
                </h1>
                <Link
                    href="/pc/projects/new"
                    className="h-9 px-4 bg-eng-blue text-white rounded text-sm font-medium flex items-center gap-1 hover:bg-blue-600"
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                        <path d="M12 4v16m8-8H4" />
                    </svg>
                    新建项目
                </Link>
            </div>

            {/* Search */}
            <div className="flex gap-3">
                <input
                    type="text"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="搜索项目名称"
                    className="flex-1 max-w-xs h-9 px-3 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                />
            </div>

            {/* Table */}
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-gray-100 bg-gray-50">
                            <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                项目名称
                            </th>
                            <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                所属部门
                            </th>
                            <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                班组长
                            </th>
                            <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                工期
                            </th>
                            <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                进度
                            </th>
                            <th className="text-left px-4 py-3 text-gray-500 font-medium">
                                状态
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {filtered.map((project) => {
                            const leader = data.users.find(
                                (u) => u.id === project.teamLeaderId
                            );
                            const dept = data.departments.find(
                                (d) => d.id === project.departmentId
                            );
                            const overallProgress =
                                project.workItems.reduce((sum, wi) => {
                                    const cumulative = getCumulativeProgress(
                                        project.id,
                                        wi.id
                                    );
                                    const pct = Math.min(
                                        cumulative / wi.targetQuantity,
                                        1
                                    );
                                    return sum + pct * wi.weight;
                                }, 0);

                            return (
                                <tr
                                    key={project.id}
                                    className="border-b border-gray-50 hover:bg-gray-50 cursor-pointer"
                                    onClick={() => router.push(`/pc/projects/${project.id}`)}
                                >
                                    <td className="px-4 py-3">
                                        <span className="text-gray-800 font-medium">
                                            {project.name}
                                        </span>
                                        {project.isSubcontractor && (
                                            <span className="ml-2 text-xs px-1.5 py-0.5 bg-purple-50 text-purple-600 rounded">
                                                分包
                                            </span>
                                        )}
                                    </td>
                                    <td className="px-4 py-3 text-gray-600">
                                        {dept?.name || "-"}
                                    </td>
                                    <td className="px-4 py-3 text-gray-600">
                                        {leader?.name || "-"}
                                    </td>
                                    <td className="px-4 py-3 text-gray-500 text-xs">
                                        {project.startDate} ~{" "}
                                        {project.endDate}
                                    </td>
                                    <td className="px-4 py-3">
                                        <div className="flex items-center gap-2">
                                            <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden max-w-[100px]">
                                                <div
                                                    className="h-full bg-eng-blue rounded-full"
                                                    style={{
                                                        width: `${Math.min(overallProgress * 100, 100)}%`,
                                                    }}
                                                />
                                            </div>
                                            <span className="text-xs text-gray-500 w-12 text-right">
                                                {(overallProgress * 100).toFixed(1)}%
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
    );
}
