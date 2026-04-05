"use client";

import { useState, useMemo, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useMockData } from "@/lib/MockDataProvider";
import { PhotoCapture } from "@/components/PhotoCapture";
import { ProjectSelector } from "@/components/ProjectSelector";

interface WorkItemInput {
    workItemId: number;
    name: string;
    targetQuantity: number;
    unit: string;
    weight: number;
    todayQuantity: number;
}

export default function ProgressPage() {
    const router = useRouter();
    const { currentUser, getProjectsByTeamLeader, getCumulativeProgress, addDailyRecord, data } =
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

    const initialItems: WorkItemInput[] = useMemo(
        () =>
            currentProject
                ? currentProject.workItems.map((wi) => ({
                      workItemId: wi.id,
                      name: wi.name,
                      targetQuantity: wi.targetQuantity,
                      unit: wi.unit,
                      weight: wi.weight,
                      todayQuantity: 0,
                  }))
                : [],
        [currentProject]
    );

    const [items, setItems] = useState<WorkItemInput[]>([]);
    const [photos, setPhotos] = useState<string[]>([]);

    // Fix: sync items when project changes
    useEffect(() => {
        setItems(initialItems);
        setPhotos([]);
    }, [currentProject?.id]); // eslint-disable-line react-hooks/exhaustive-deps

    const handlePhotosChange = useCallback((newPhotos: string[]) => {
        setPhotos(newPhotos);
    }, []);

    if (!currentUser) return null;

    const updateQuantity = (workItemId: number, value: string) => {
        const num = parseFloat(value) || 0;
        setItems((prev) =>
            prev.map((item) =>
                item.workItemId === workItemId
                    ? { ...item, todayQuantity: num }
                    : item
            )
        );
    };

    const overallProgress = items.reduce((sum, item) => {
        if (!currentProject) return sum;
        const cumulative = getCumulativeProgress(
            currentProject.id,
            item.workItemId
        );
        const total = cumulative + item.todayQuantity;
        const pct = Math.min(total / item.targetQuantity, 1);
        return sum + pct * item.weight;
    }, 0);

    const handleSubmit = () => {
        if (!currentProject || !currentUser) return;
        const progressData = items
            .filter((i) => i.todayQuantity > 0)
            .map((i, idx) => ({
                id: 100 + idx,
                dailyRecordId: 0,
                workItemId: i.workItemId,
                completedQuantity: i.todayQuantity,
                photoUrls: photos,
            }));
        if (progressData.length === 0) return;
        addDailyRecord({
            projectId: currentProject.id,
            teamLeaderId: currentUser.id,
            date: new Date().toISOString().split("T")[0],
            moduleType: "进度",
            status: "待审核",
            reviewComment: null,
            reviewedAt: null,
            progress: progressData,
        });
        router.push("/home");
    };

    const hasInput = items.some((i) => i.todayQuantity > 0);

    return (
        <div className="flex flex-col min-h-screen">
            {/* Header */}
            <div className="px-4 pt-4 pb-2">
                <div className="flex items-center gap-2 mb-2">
                    <button onClick={() => router.back()} className="text-eng-gray-500">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                            <path d="M15 19l-7-7 7-7" />
                        </svg>
                    </button>
                    <h1 className="text-app-title font-bold text-eng-gray-900">
                        施工进度
                    </h1>
                </div>

                {/* Project selector for non-班组长 */}
                {!isTeamLeader && (
                    <div className="mb-2">
                        <ProjectSelector
                            value={selectedProjectId}
                            onChange={(id) => setSelectedProjectId(id)}
                            className="w-full"
                        />
                    </div>
                )}

                {currentProject && (
                    <p className="text-xs text-eng-gray-400">
                        {currentProject.name}
                    </p>
                )}
            </div>

            {!currentProject ? (
                <div className="flex-1 flex items-center justify-center px-4">
                    <p className="text-sm text-gray-400">
                        {isTeamLeader ? "暂无负责的项目" : "请选择一个项目"}
                    </p>
                </div>
            ) : (
                <>
                    {/* Overall Progress Bar */}
                    <div className="px-4 py-3 bg-white border-y border-eng-gray-100">
                        <div className="flex items-center justify-between mb-1">
                            <span className="text-xs text-eng-gray-500">综合进度</span>
                            <span className="text-sm font-bold text-eng-blue">
                                {(overallProgress * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="h-2 bg-eng-gray-100 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-eng-blue rounded-full transition-all"
                                style={{ width: `${Math.min(overallProgress * 100, 100)}%` }}
                            />
                        </div>
                    </div>

                    {/* Work Items */}
                    <div className="flex-1 px-4 py-3 space-y-3">
                        {items.map((item) => {
                            const cumulative = getCumulativeProgress(currentProject.id, item.workItemId);
                            const total = cumulative + item.todayQuantity;
                            const pct = Math.min((total / item.targetQuantity) * 100, 100);

                            return (
                                <div
                                    key={item.workItemId}
                                    className="bg-white rounded-card border border-eng-gray-100 p-3"
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm font-medium text-eng-gray-800">
                                            {item.name}
                                        </span>
                                        <span className="text-xs text-eng-gray-400">
                                            权重 {(item.weight * 100).toFixed(0)}%
                                        </span>
                                    </div>

                                    <div className="h-1.5 bg-eng-gray-100 rounded-full overflow-hidden mb-2">
                                        <div
                                            className={`h-full rounded-full transition-all ${
                                                pct >= 100
                                                    ? "bg-eng-green"
                                                    : "bg-eng-blue"
                                            }`}
                                            style={{ width: `${pct}%` }}
                                        />
                                    </div>

                                    <div className="flex items-center justify-between text-xs text-eng-gray-400 mb-2">
                                        <span>
                                            累计 {cumulative} / 目标 {item.targetQuantity} {item.unit}
                                        </span>
                                        <span>{pct.toFixed(1)}%</span>
                                    </div>

                                    <div className="flex items-center gap-2">
                                        <label className="text-xs text-eng-gray-500">
                                            本次完成
                                        </label>
                                        <input
                                            type="number"
                                            min="0"
                                            value={item.todayQuantity || ""}
                                            onChange={(e) =>
                                                updateQuantity(
                                                    item.workItemId,
                                                    e.target.value
                                                )
                                            }
                                            placeholder="0"
                                            className="flex-1 h-9 px-2 border border-eng-gray-200 rounded text-sm text-right focus:outline-none focus:border-eng-blue"
                                        />
                                        <span className="text-xs text-eng-gray-400">
                                            {item.unit}
                                        </span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Photo + Submit */}
                    <div className="px-4 py-3 bg-white border-t border-eng-gray-100 space-y-3">
                        <PhotoCapture
                            photos={photos}
                            onChange={handlePhotosChange}
                            max={5}
                            label="进度照片"
                        />

                        <button
                            onClick={handleSubmit}
                            disabled={!hasInput}
                            className={`w-full h-11 rounded-btn text-sm font-medium transition-colors ${
                                hasInput
                                    ? "bg-eng-blue text-white hover:bg-blue-600"
                                    : "bg-eng-gray-200 text-eng-gray-400 cursor-not-allowed"
                            }`}
                        >
                            提交审核
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}
