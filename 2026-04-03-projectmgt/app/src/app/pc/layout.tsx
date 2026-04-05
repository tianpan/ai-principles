"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useMockData } from "@/lib/MockDataProvider";
import { AiQueryPanel } from "@/components/ai/AiQueryPanel";
import { executeNLQuery, type NLQueryResult, type QueryDataContext } from "@/lib/ai/nl-query";
import Link from "next/link";

const menuItems = [
    {
        role: "项目经理",
        items: [
            { href: "/pc/dashboard", label: "工作台", icon: DashboardIcon },
            { href: "/pc/projects", label: "项目管理", icon: ProjectIcon },
            { href: "/pc/review", label: "审核工作台", icon: ReviewIcon },
            { href: "/pc/reports", label: "报表中心", icon: ReportIcon },
        ],
    },
    {
        role: "管理员",
        items: [
            { href: "/pc/dashboard", label: "工作台", icon: DashboardIcon },
            { href: "/pc/projects", label: "项目管理", icon: ProjectIcon },
            { href: "/pc/review", label: "审核工作台", icon: ReviewIcon },
            { href: "/pc/reports", label: "报表中心", icon: ReportIcon },
        ],
    },
];

function PcShell({ children }: { children: React.ReactNode }) {
    const { currentUser, logout, hydrated, data, getCumulativeProgress } = useMockData();
    const pathname = usePathname();
    const router = useRouter();
    const [queryInput, setQueryInput] = useState("");
    const [queryResult, setQueryResult] = useState<NLQueryResult | null>(null);

    const queryContext: QueryDataContext = useMemo(() => {
        const project = data.projects[0];
        const today = new Date().toISOString().split("T")[0];
        const todayRecords = data.dailyRecords.filter(
            (r) => r.projectId === project?.id && r.date === today
        );
        return {
            projectName: project?.name ?? "",
            workItems: project?.workItems.map((wi) => ({
                name: wi.name,
                targetQuantity: wi.targetQuantity,
                completedQuantity: project ? getCumulativeProgress(project.id, wi.id) : 0,
                unit: wi.unit,
            })) ?? [],
            todayAttendance: todayRecords
                .filter((r) => r.moduleType === "考勤")
                .reduce((sum, r) => {
                    const att = r.attendance as Record<string, unknown> | undefined;
                    return sum + ((att?.attendeeCount as number) ?? 0);
                }, 0),
            materialSummary: todayRecords
                .filter((r) => r.moduleType === "材料")
                .flatMap((r) => {
                    const mats = r.materials as Array<Record<string, unknown>> | undefined;
                    return mats
                        ? mats.map((m) => ({
                            name: (m.name as string) ?? "",
                            type: (m.recordType as string) ?? "领料",
                            quantity: (m.quantity as number) ?? 0,
                            unit: (m.unit as string) ?? "",
                        }))
                        : [];
                }),
            totalRecords: todayRecords.length,
            approvedRecords: todayRecords.filter((r) => r.status === "已通过").length,
        };
    }, [data.projects, data.dailyRecords, getCumulativeProgress]);

    const handleQuery = useCallback(() => {
        if (!queryInput.trim()) return;
        const result = executeNLQuery(queryInput.trim(), queryContext);
        setQueryResult(result);
    }, [queryInput, queryContext]);

    useEffect(() => {
        if (hydrated && !currentUser) {
            router.replace("/login");
        }
    }, [hydrated, currentUser, router]);

    if (!hydrated) {
        return null;
    }

    if (!currentUser) {
        return null;
    }

    const roleMenu =
        menuItems.find((m) => m.role === currentUser?.role)?.items ||
        menuItems[0].items;

    return (
        <div className="flex h-screen bg-gray-100">
            {/* Sidebar */}
            <aside className="w-sidebar bg-white border-r border-gray-200 flex flex-col shrink-0">
                <div className="h-14 flex items-center px-5 border-b border-gray-100">
                    <h1 className="text-base font-bold text-gray-800">
                        工程管家
                    </h1>
                    <span className="ml-2 text-xs text-gray-400">管理端</span>
                </div>

                <nav className="flex-1 py-2">
                    {roleMenu.map((item) => (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`flex items-center gap-3 px-5 py-2.5 text-sm transition-colors ${
                                pathname === item.href ||
                                pathname.startsWith(item.href + "/")
                                    ? "bg-blue-50 text-eng-blue font-medium"
                                    : "text-gray-600 hover:bg-gray-50"
                            }`}
                        >
                            <item.icon
                                active={
                                    pathname === item.href ||
                                    pathname.startsWith(item.href + "/")
                                }
                            />
                            {item.label}
                        </Link>
                    ))}
                </nav>

                <div className="border-t border-gray-100 p-4">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-eng-blue rounded-full flex items-center justify-center text-white text-sm">
                            {currentUser?.name[0]}
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm text-gray-800 truncate">
                                {currentUser?.name}
                            </p>
                            <p className="text-xs text-gray-400">
                                {currentUser?.role}
                            </p>
                        </div>
                        <button
                            onClick={() => {
                                logout();
                                router.push("/login");
                            }}
                            className="text-gray-400 hover:text-gray-600"
                            title="退出登录"
                        >
                            <svg
                                width="18"
                                height="18"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth={1.5}
                            >
                                <path d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                            </svg>
                        </button>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-auto">
                {/* NL 查询栏 */}
                <div className="border-b border-gray-100 bg-white px-6 py-3">
                    <div className="flex items-center gap-2">
                        <div className="flex-1 relative">
                            <input
                                type="text"
                                value={queryInput}
                                onChange={(e) => setQueryInput(e.target.value)}
                                onKeyDown={(e) => e.key === "Enter" && handleQuery()}
                                placeholder='输入问题查询项目数据，如"还剩多少没焊"'
                                className="w-full h-9 pl-4 pr-10 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-eng-blue"
                            />
                            <button
                                onClick={handleQuery}
                                className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-eng-blue"
                            >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                                    <circle cx="11" cy="11" r="8" />
                                    <path d="m21 21-4.35-4.35" />
                                </svg>
                            </button>
                        </div>
                    </div>
                    {queryResult && (
                        <div className="mt-2">
                            <AiQueryPanel
                                result={queryResult}
                                onClose={() => {
                                    setQueryResult(null);
                                    setQueryInput("");
                                }}
                            />
                        </div>
                    )}
                </div>
                <div className="p-6">{children}</div>
            </main>
        </div>
    );
}

export default function PcLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return <PcShell>{children}</PcShell>;
}

function DashboardIcon({ active }: { active: boolean }) {
    return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={active ? "#1677ff" : "currentColor"} strokeWidth={1.5}>
            <path d="M4 5a1 1 0 011-1h4a1 1 0 011 1v5a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm10 0a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zm10-2a1 1 0 011-1h4a1 1 0 011 1v6a1 1 0 01-1 1h-4a1 1 0 01-1-1v-6z" />
        </svg>
    );
}

function ProjectIcon({ active }: { active: boolean }) {
    return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={active ? "#1677ff" : "currentColor"} strokeWidth={1.5}>
            <path d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
        </svg>
    );
}

function ReviewIcon({ active }: { active: boolean }) {
    return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={active ? "#1677ff" : "currentColor"} strokeWidth={1.5}>
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
        </svg>
    );
}

function ReportIcon({ active }: { active: boolean }) {
    return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={active ? "#1677ff" : "currentColor"} strokeWidth={1.5}>
            <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
    );
}
