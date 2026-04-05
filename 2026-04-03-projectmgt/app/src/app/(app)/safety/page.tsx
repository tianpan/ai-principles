"use client";

import { useState, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useMockData } from "@/lib/MockDataProvider";
import { PhotoCapture } from "@/components/PhotoCapture";
import { ProjectSelector } from "@/components/ProjectSelector";

const STEPS = ["拍照签到", "安全交底", "签名确认", "提交审核"];

export default function SafetyPage() {
    const router = useRouter();
    const { currentUser, getProjectsByTeamLeader, addDailyRecord, data } = useMockData();
    const [step, setStep] = useState(0);

    // Project selection for non-班组长
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

    // Step 1 state
    const [gpsLocation] = useState("32.06N, 118.79E");
    const [timestamp] = useState(new Date().toISOString());
    const [selectedAttendees, setSelectedAttendees] = useState<number[]>([]);
    const [photos, setPhotos] = useState<string[]>([]);

    // Step 2 state
    const [briefingId, setBriefingId] = useState<number | null>(null);

    // Step 3 state
    const [signatureData, setSignatureData] = useState<string | null>(null);

    // High-risk work
    const [highRiskWork, setHighRiskWork] = useState("");

    const projectMembers = currentProject
        ? data.workers.filter((w) =>
              currentProject.memberIds.includes(w.id)
          )
        : [];

    const selectedTemplate = data.templates.find(
        (t) => t.id === briefingId
    );

    const toggleAttendee = useCallback((id: number) => {
        setSelectedAttendees((prev) =>
            prev.includes(id)
                ? prev.filter((x) => x !== id)
                : [...prev, id]
        );
    }, []);

    const handlePhotosChange = useCallback((newPhotos: string[]) => {
        setPhotos(newPhotos);
    }, []);

    const handleSubmit = () => {
        if (!currentProject || !currentUser) return;
        addDailyRecord({
            projectId: currentProject.id,
            teamLeaderId: currentUser.id,
            date: new Date().toISOString().split("T")[0],
            moduleType: "考勤",
            status: "待审核",
            reviewComment: null,
            reviewedAt: null,
            attendance: {
                photoUrls: photos,
                gpsLocation,
                safetyBriefingType: selectedTemplate?.category || "通用",
                safetyBriefingId: briefingId,
                signatureUrl: signatureData || "/placeholder/signature.png",
                attendeeIds: selectedAttendees,
                highRiskWork: highRiskWork || null,
            },
        });
        router.push("/home");
    };

    const canNext = useMemo(() => {
        if (!currentProject) return false;
        switch (step) {
            case 0:
                return photos.length > 0 && selectedAttendees.length > 0;
            case 1:
                return briefingId !== null;
            case 2:
                return true;
            case 3:
                return true;
            default:
                return false;
        }
    }, [step, photos, selectedAttendees, briefingId, currentProject]);

    if (!currentUser) return null;

    return (
        <div className="flex flex-col min-h-screen">
            {/* Header */}
            <div className="px-4 pt-4 pb-2">
                <div className="flex items-center gap-2 mb-3">
                    <button
                        onClick={() => step > 0 ? setStep(step - 1) : router.back()}
                        className="text-eng-gray-500"
                    >
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                            <path d="M15 19l-7-7 7-7" />
                        </svg>
                    </button>
                    <h1 className="text-app-title font-bold text-eng-gray-900">
                        安全考勤
                    </h1>
                </div>

                {/* Project selector for non-班组长 */}
                {!isTeamLeader && (
                    <div className="mb-3">
                        <ProjectSelector
                            value={selectedProjectId}
                            onChange={(id) => {
                                setSelectedProjectId(id);
                                setSelectedAttendees([]);
                            }}
                            className="w-full"
                        />
                    </div>
                )}

                {/* Step Indicator */}
                <div className="flex items-center justify-between mb-1">
                    {STEPS.map((s, i) => (
                        <div key={s} className="flex items-center">
                            <div
                                className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium ${
                                    i < step
                                        ? "bg-eng-green text-white"
                                        : i === step
                                        ? "bg-eng-blue text-white"
                                        : "bg-eng-gray-200 text-eng-gray-400"
                                }`}
                            >
                                {i < step ? "✓" : i + 1}
                            </div>
                            {i < STEPS.length - 1 && (
                                <div
                                    className={`w-10 sm:w-16 h-0.5 ${
                                        i < step ? "bg-eng-green" : "bg-eng-gray-200"
                                    }`}
                                />
                            )}
                        </div>
                    ))}
                </div>
                <p className="text-xs text-eng-gray-400 text-center">
                    {STEPS[step]}
                </p>
            </div>

            {/* No project message */}
            {!currentProject && (
                <div className="flex-1 flex items-center justify-center px-4">
                    <p className="text-sm text-gray-400">
                        {isTeamLeader ? "暂无负责的项目" : "请选择一个项目"}
                    </p>
                </div>
            )}

            {/* Step Content */}
            {currentProject && (
                <div className="flex-1 px-4 py-4">
                    {step === 0 && (
                        <StepPhoto
                            gpsLocation={gpsLocation}
                            timestamp={timestamp}
                            members={projectMembers}
                            selected={selectedAttendees}
                            onToggle={toggleAttendee}
                            photos={photos}
                            onPhotosChange={handlePhotosChange}
                        />
                    )}
                    {step === 1 && (
                        <StepBriefing
                            templates={data.templates}
                            selectedId={briefingId}
                            onSelect={setBriefingId}
                        />
                    )}
                    {step === 2 && (
                        <StepSignature
                            signatureData={signatureData}
                            onSign={setSignatureData}
                        />
                    )}
                    {step === 3 && (
                        <StepConfirm
                            attendees={projectMembers.filter((w) =>
                                selectedAttendees.includes(w.id)
                            )}
                            template={selectedTemplate}
                            hasSignature={!!signatureData}
                            highRiskWork={highRiskWork}
                            onHighRiskChange={setHighRiskWork}
                        />
                    )}
                </div>
            )}

            {/* Footer Navigation */}
            {currentProject && (
                <div className="px-4 py-3 bg-white border-t border-eng-gray-100 flex gap-3">
                    {step > 0 && (
                        <button
                            onClick={() => setStep(step - 1)}
                            className="flex-1 h-11 border border-eng-gray-200 rounded-btn text-sm text-eng-gray-600"
                        >
                            上一步
                        </button>
                    )}
                    <button
                        onClick={() => {
                            if (step < 3) setStep(step + 1);
                            else handleSubmit();
                        }}
                        disabled={!canNext}
                        className={`flex-1 h-11 rounded-btn text-sm font-medium transition-colors ${
                            canNext
                                ? "bg-eng-blue text-white hover:bg-blue-600"
                                : "bg-eng-gray-200 text-eng-gray-400 cursor-not-allowed"
                        }`}
                    >
                        {step < 3 ? "下一步" : "提交审核"}
                    </button>
                </div>
            )}
        </div>
    );
}

/* ---------- Step 1: Photo + GPS + Attendees ---------- */
function StepPhoto({
    gpsLocation,
    timestamp,
    members,
    selected,
    onToggle,
    photos,
    onPhotosChange,
}: {
    gpsLocation: string;
    timestamp: string;
    members: Array<{ id: number; name: string; position: string }>;
    selected: number[];
    onToggle: (id: number) => void;
    photos: string[];
    onPhotosChange: (photos: string[]) => void;
}) {
    return (
        <div className="space-y-4">
            {/* Photo Capture */}
            <PhotoCapture
                photos={photos}
                onChange={onPhotosChange}
                max={3}
                label="班前会照片"
            />

            {/* GPS + Time */}
            <div className="flex gap-3">
                <div className="flex-1 bg-eng-gray-50 rounded-card p-3">
                    <p className="text-xs text-eng-gray-400">GPS定位</p>
                    <p className="text-sm text-eng-gray-700">{gpsLocation}</p>
                </div>
                <div className="flex-1 bg-eng-gray-50 rounded-card p-3">
                    <p className="text-xs text-eng-gray-400">签到时间</p>
                    <p className="text-sm text-eng-gray-700">
                        {new Date(timestamp).toLocaleString("zh-CN")}
                    </p>
                </div>
            </div>

            {/* Attendee List */}
            <div>
                <h3 className="text-sm font-medium text-eng-gray-700 mb-2">
                    参会人员 ({selected.length}/{members.length})
                </h3>
                {members.length > 0 ? (
                    <div className="space-y-2">
                        {members.map((m) => (
                            <label
                                key={m.id}
                                className={`flex items-center gap-3 p-3 rounded-card border cursor-pointer transition-colors ${
                                    selected.includes(m.id)
                                        ? "border-eng-blue bg-eng-blue/5"
                                        : "border-eng-gray-100 bg-white"
                                }`}
                            >
                                <input
                                    type="checkbox"
                                    checked={selected.includes(m.id)}
                                    onChange={() => onToggle(m.id)}
                                    className="w-4 h-4 accent-eng-blue"
                                />
                                <span className="text-sm text-eng-gray-800">
                                    {m.name}
                                </span>
                                <span className="text-xs text-eng-gray-400 ml-auto">
                                    {m.position}
                                </span>
                            </label>
                        ))}
                    </div>
                ) : (
                    <p className="text-sm text-gray-400 py-4 text-center">
                        该项目暂无成员
                    </p>
                )}
            </div>
        </div>
    );
}

/* ---------- Step 2: Safety Briefing ---------- */
function StepBriefing({
    templates,
    selectedId,
    onSelect,
}: {
    templates: Array<{ id: number; name: string; category: string; content: string }>;
    selectedId: number | null;
    onSelect: (id: number) => void;
}) {
    return (
        <div className="space-y-3">
            <h3 className="text-sm font-medium text-eng-gray-700">
                选择安全交底模板
            </h3>
            {templates.map((t) => (
                <label
                    key={t.id}
                    className={`block p-4 rounded-card border cursor-pointer transition-colors ${
                        selectedId === t.id
                            ? "border-eng-blue bg-eng-blue/5"
                            : "border-eng-gray-100 bg-white"
                    }`}
                >
                    <div className="flex items-center gap-2 mb-2">
                        <input
                            type="radio"
                            name="briefing"
                            checked={selectedId === t.id}
                            onChange={() => onSelect(t.id)}
                            className="accent-eng-blue"
                        />
                        <span className="text-sm font-medium text-eng-gray-800">
                            {t.name}
                        </span>
                        <span className="text-xs px-1.5 py-0.5 bg-eng-gray-100 rounded text-eng-gray-500">
                            {t.category}
                        </span>
                    </div>
                    {selectedId === t.id && (
                        <div className="mt-2 pl-6 text-xs text-eng-gray-600 whitespace-pre-line leading-relaxed">
                            {t.content}
                        </div>
                    )}
                </label>
            ))}
        </div>
    );
}

/* ---------- Step 3: Signature ---------- */
function StepSignature({
    signatureData,
    onSign,
}: {
    signatureData: string | null;
    onSign: (data: string | null) => void;
}) {
    return (
        <div className="space-y-4">
            <h3 className="text-sm font-medium text-eng-gray-700">
                班组长签名确认
            </h3>
            <div className="bg-white border-2 border-dashed border-eng-gray-300 rounded-card h-48 flex items-center justify-center">
                {signatureData ? (
                    <div className="text-center">
                        <div className="text-eng-green text-4xl mb-2">✓</div>
                        <p className="text-sm text-eng-green">已签名</p>
                    </div>
                ) : (
                    <p className="text-eng-gray-400 text-sm">
                        点击下方按钮模拟签名
                    </p>
                )}
            </div>
            <div className="flex gap-3">
                <button
                    onClick={() => onSign(null)}
                    className="flex-1 h-10 border border-eng-gray-200 rounded-btn text-sm text-eng-gray-600"
                >
                    清除
                </button>
                <button
                    onClick={() => onSign("mock-signature-data")}
                    className="flex-1 h-10 bg-eng-blue text-white rounded-btn text-sm font-medium"
                >
                    确认签名
                </button>
            </div>
        </div>
    );
}

/* ---------- Step 4: Confirm ---------- */
function StepConfirm({
    attendees,
    template,
    hasSignature,
    highRiskWork,
    onHighRiskChange,
}: {
    attendees: Array<{ name: string; position: string }>;
    template: { name: string; category: string } | undefined;
    hasSignature: boolean;
    highRiskWork: string;
    onHighRiskChange: (v: string) => void;
}) {
    return (
        <div className="space-y-4">
            <h3 className="text-sm font-medium text-eng-gray-700">
                确认提交信息
            </h3>

            <div className="bg-white rounded-card border border-eng-gray-100 p-4 space-y-3">
                <InfoRow
                    label="参会人数"
                    value={`${attendees.length} 人`}
                />
                <InfoRow
                    label="参会人员"
                    value={attendees.length > 0 ? attendees.map((a) => a.name).join("、") : "无"}
                />
                <InfoRow
                    label="安全交底"
                    value={template?.name || "未选择"}
                />
                <InfoRow
                    label="签名状态"
                    value={hasSignature ? "已签名" : "未签名"}
                />
            </div>

            {/* High-risk work */}
            <div>
                <label className="block text-sm text-eng-gray-700 mb-1">
                    高风险作业（选填）
                </label>
                <textarea
                    value={highRiskWork}
                    onChange={(e) => onHighRiskChange(e.target.value)}
                    placeholder="如有高风险作业请描述，如：深基坑开挖、高空作业等"
                    className="w-full h-20 px-3 py-2 border border-eng-gray-200 rounded-btn text-sm resize-none focus:outline-none focus:border-eng-blue"
                />
            </div>
        </div>
    );
}

function InfoRow({ label, value }: { label: string; value: string }) {
    return (
        <div className="flex items-start gap-2">
            <span className="text-xs text-eng-gray-400 shrink-0 w-16">
                {label}
            </span>
            <span className="text-sm text-eng-gray-800">{value}</span>
        </div>
    );
}
