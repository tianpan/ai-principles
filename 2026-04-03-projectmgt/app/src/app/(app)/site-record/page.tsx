"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useMockData } from "@/lib/MockDataProvider";
import { ProjectSelector } from "@/components/ProjectSelector";
import { PhotoCapture } from "@/components/PhotoCapture";

const recordTypes = [
    "管线安装",
    "管沟开挖",
    "阀门安装",
    "管沟回填",
    "吹扫试压",
    "入户改造",
    "旧管拆除",
];

interface FormField {
    label: string;
    key: string;
    unit?: string;
    type?: "text" | "number";
}

const formFieldsMap: Record<string, FormField[]> = {
    管线安装: [
        { label: "位置", key: "location" },
        { label: "管径", key: "diameter" },
        { label: "材质", key: "material" },
        { label: "长度", key: "length", unit: "米", type: "number" },
        { label: "深度", key: "depth", unit: "米", type: "number" },
        { label: "宽度", key: "width", unit: "米", type: "number" },
    ],
    管沟开挖: [
        { label: "位置", key: "location" },
        { label: "长度", key: "length", unit: "米", type: "number" },
        { label: "深度", key: "depth", unit: "米", type: "number" },
        { label: "宽度", key: "width", unit: "米", type: "number" },
    ],
    阀门安装: [
        { label: "位置", key: "location" },
        { label: "规格", key: "specification" },
        { label: "数量", key: "quantity", unit: "个", type: "number" },
    ],
    管沟回填: [
        { label: "位置", key: "location" },
        { label: "长度", key: "length", unit: "米", type: "number" },
        { label: "回填材料", key: "backfillMaterial" },
    ],
    吹扫试压: [
        { label: "试验类型", key: "testType" },
        { label: "压力值", key: "pressure", unit: "MPa", type: "number" },
        { label: "持续时间", key: "duration", unit: "分钟", type: "number" },
    ],
    入户改造: [
        { label: "楼栋号", key: "building" },
        { label: "户数", key: "households", unit: "户", type: "number" },
        { label: "管材", key: "material" },
    ],
    旧管拆除: [
        { label: "位置", key: "location" },
        { label: "管径", key: "diameter" },
        { label: "拆除长度", key: "length", unit: "米", type: "number" },
    ],
};

export default function SiteRecordPage() {
    const router = useRouter();
    const { currentUser, getProjectsByTeamLeader, addDailyRecord, data } = useMockData();

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

    const [recordType, setRecordType] = useState("管线安装");
    const [formData, setFormData] = useState<Record<string, string | number>>({});
    const [isHiddenWork, setIsHiddenWork] = useState(false);
    const [hiddenWorkDesc, setHiddenWorkDesc] = useState("");
    const [photos, setPhotos] = useState<string[]>([]);
    const [sketchPhotos, setSketchPhotos] = useState<string[]>([]);
    const [validationMsg, setValidationMsg] = useState("");

    const fields = formFieldsMap[recordType] || [];

    const updateField = (key: string, value: string | number) => {
        setFormData((prev) => ({ ...prev, [key]: value }));
    };

    const handlePhotosChange = useCallback((newPhotos: string[]) => {
        setPhotos(newPhotos);
    }, []);

    const handleSketchPhotosChange = useCallback((newPhotos: string[]) => {
        setSketchPhotos(newPhotos);
    }, []);

    // Validation: location field is required for most record types
    const hasRequiredFields = (() => {
        if (recordType === "吹扫试压") {
            return !!formData["testType"];
        }
        if (recordType === "入户改造") {
            return !!formData["building"];
        }
        return !!formData["location"];
    })();

    const handleSubmit = () => {
        if (!currentProject || !currentUser) return;
        setValidationMsg("");

        if (!hasRequiredFields) {
            const fieldLabel = recordType === "吹扫试压" ? "试验类型" :
                recordType === "入户改造" ? "楼栋号" : "位置";
            setValidationMsg(`请填写${fieldLabel}`);
            return;
        }

        addDailyRecord({
            projectId: currentProject.id,
            teamLeaderId: currentUser.id,
            date: new Date().toISOString().split("T")[0],
            moduleType: "施工记录",
            status: "待审核",
            reviewComment: null,
            reviewedAt: null,
            siteRecord: {
                formData: { recordType, ...formData },
                photoUrls: photos,
                sketchPhotoUrls: sketchPhotos,
                pdfAnnotation: null,
                hiddenWork: isHiddenWork,
                hiddenWorkDescription: isHiddenWork ? hiddenWorkDesc : null,
            },
        });
        router.push("/home");
    };

    if (!currentUser) return null;

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
                        施工记录
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
                    {/* Content */}
                    <div className="flex-1 px-4 py-3 space-y-4">
                        {/* Record Type Selector */}
                        <div>
                            <label className="block text-sm text-eng-gray-700 mb-1.5">
                                记录类型
                            </label>
                            <div className="flex flex-wrap gap-2">
                                {recordTypes.map((rt) => (
                                    <button
                                        key={rt}
                                        onClick={() => {
                                            setRecordType(rt);
                                            setFormData({});
                                            setValidationMsg("");
                                        }}
                                        className={`px-3 py-1.5 rounded-full text-xs transition-colors ${
                                            recordType === rt
                                                ? "bg-eng-blue text-white"
                                                : "bg-eng-gray-100 text-eng-gray-600"
                                        }`}
                                    >
                                        {rt}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Dynamic Form Fields */}
                        <div className="space-y-3">
                            {fields.map((field) => (
                                <div key={field.key}>
                                    <label className="block text-sm text-eng-gray-600 mb-1">
                                        {field.label}
                                        {field.key === "location" && (
                                            <span className="text-eng-red"> *</span>
                                        )}
                                    </label>
                                    <div className="flex items-center gap-2">
                                        <input
                                            type={field.type || "text"}
                                            value={formData[field.key] ?? ""}
                                            onChange={(e) =>
                                                updateField(
                                                    field.key,
                                                    field.type === "number"
                                                        ? parseFloat(e.target.value) || 0
                                                        : e.target.value
                                                )
                                            }
                                            placeholder={`请输入${field.label}`}
                                            className="flex-1 h-10 px-3 border border-eng-gray-200 rounded-btn text-sm focus:outline-none focus:border-eng-blue"
                                        />
                                        {field.unit && (
                                            <span className="text-xs text-eng-gray-400 shrink-0">
                                                {field.unit}
                                            </span>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Hidden Work */}
                        <div className="bg-white rounded-card border border-eng-gray-100 p-3">
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={isHiddenWork}
                                    onChange={(e) => setIsHiddenWork(e.target.checked)}
                                    className="w-4 h-4 accent-eng-blue"
                                />
                                <span className="text-sm text-eng-gray-700">
                                    隐蔽工程
                                </span>
                            </label>
                            {isHiddenWork && (
                                <textarea
                                    value={hiddenWorkDesc}
                                    onChange={(e) => setHiddenWorkDesc(e.target.value)}
                                    placeholder="请描述隐蔽工程内容"
                                    className="w-full h-20 px-3 py-2 border border-eng-gray-200 rounded-btn text-sm resize-none mt-2 focus:outline-none focus:border-eng-blue"
                                />
                            )}
                        </div>

                        {/* Photo Upload */}
                        <PhotoCapture
                            photos={photos}
                            onChange={handlePhotosChange}
                            max={10}
                            label="施工照片"
                        />

                        {/* Sketch Photos */}
                        <PhotoCapture
                            photos={sketchPhotos}
                            onChange={handleSketchPhotosChange}
                            max={5}
                            label="完工草图 / 竣工图"
                        />
                    </div>

                    {/* Submit */}
                    <div className="px-4 py-3 bg-white border-t border-eng-gray-100 space-y-2">
                        {validationMsg && (
                            <p className="text-xs text-red-500">{validationMsg}</p>
                        )}
                        <button
                            onClick={handleSubmit}
                            disabled={!hasRequiredFields || !currentProject}
                            className={`w-full h-11 rounded-btn text-sm font-medium transition-colors ${
                                hasRequiredFields && currentProject
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
