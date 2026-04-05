"use client";

import { useState, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useMockData } from "@/lib/MockDataProvider";
import { ProjectSelector } from "@/components/ProjectSelector";
import { MaterialQuickSelect, generateMaterialNo } from "@/components/MaterialQuickSelect";

const TABS = [
    { key: "领料", label: "领料" },
    { key: "退料", label: "退料" },
    { key: "设备", label: "设备" },
    { key: "采买", label: "采买" },
];

const EQUIPMENT_PRESETS = [
    "吊车",
    "挖掘机",
    "装载机",
    "压路机",
    "发电机",
    "电焊机",
    "切割机",
    "空压机",
    "水泵",
    "混凝土搅拌机",
];

interface MaterialRow {
    id: number;
    requisitionNo: string;
    materialName: string;
    specification: string;
    quantity: number;
}

interface EquipmentRow {
    id: number;
    equipmentType: string;
    specification: string;
    shifts: number;
}

export default function MaterialPage() {
    const router = useRouter();
    const { currentUser, getProjectsByTeamLeader, addDailyRecord, getDailyRecordsByProject, data } =
        useMockData();
    const [activeTab, setActiveTab] = useState("领料");
    const [validationMsg, setValidationMsg] = useState("");

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

    // Material rows (领料/退料/采买)
    const [materialRows, setMaterialRows] = useState<MaterialRow[]>([
        { id: 1, requisitionNo: "", materialName: "", specification: "", quantity: 0 },
    ]);

    // Equipment rows (设备)
    const [equipmentRows, setEquipmentRows] = useState<EquipmentRow[]>([
        { id: 1, equipmentType: "", specification: "", shifts: 0 },
    ]);

    const isEquipment = activeTab === "设备";

    const addMaterialRow = () => {
        setMaterialRows((prev) => [
            ...prev,
            {
                id: Date.now(),
                requisitionNo: activeTab === "采买" ? "" : generateMaterialNo(),
                materialName: "",
                specification: "",
                quantity: 0,
            },
        ]);
    };

    const removeMaterialRow = (id: number) => {
        setMaterialRows((prev) => prev.filter((r) => r.id !== id));
    };

    const updateMaterialRow = (
        id: number,
        field: keyof MaterialRow,
        value: string | number
    ) => {
        setMaterialRows((prev) =>
            prev.map((r) => (r.id === id ? { ...r, [field]: value } : r))
        );
    };

    const handleQuickSelect = useCallback(
        (item: { name: string; unit: string }) => {
            setMaterialRows((prev) => [
                ...prev,
                {
                    id: Date.now(),
                    requisitionNo: activeTab === "采买" ? "" : generateMaterialNo(),
                    materialName: item.name,
                    specification: item.unit,
                    quantity: 1,
                },
            ]);
        },
        [activeTab]
    );

    const addEquipmentRow = () => {
        setEquipmentRows((prev) => [
            ...prev,
            { id: Date.now(), equipmentType: "", specification: "", shifts: 0 },
        ]);
    };

    const removeEquipmentRow = (id: number) => {
        setEquipmentRows((prev) => prev.filter((r) => r.id !== id));
    };

    const updateEquipmentRow = (
        id: number,
        field: keyof EquipmentRow,
        value: string | number
    ) => {
        setEquipmentRows((prev) =>
            prev.map((r) => (r.id === id ? { ...r, [field]: value } : r))
        );
    };

    // Copy yesterday
    const handleCopyYesterday = () => {
        if (!currentProject) return;
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        const yesterdayStr = yesterday.toISOString().split("T")[0];
        const yesterdayRecords = getDailyRecordsByProject(
            currentProject.id
        ).filter(
            (r) =>
                r.date === yesterdayStr &&
                r.moduleType === "材料" &&
                r.status !== "已退回"
        );
        if (yesterdayRecords.length === 0) return;
        const lastRecord = yesterdayRecords[0];
        const materials = lastRecord.materials as
            | Array<{
                  requisitionNo: string | null;
                  materialName: string;
                  specification: string;
                  quantity: number;
              }>
            | undefined;
        if (materials && materials.length > 0) {
            setMaterialRows(
                materials.map((m, i) => ({
                    id: Date.now() + i,
                    requisitionNo: m.requisitionNo || "",
                    materialName: m.materialName,
                    specification: m.specification,
                    quantity: m.quantity,
                }))
            );
        }
    };

    const totalQuantity = useMemo(
        () =>
            isEquipment
                ? equipmentRows.reduce((s, r) => s + r.shifts, 0)
                : materialRows.reduce((s, r) => s + (r.quantity || 0), 0),
        [isEquipment, materialRows, equipmentRows]
    );

    const totalRows = isEquipment ? equipmentRows.length : materialRows.length;

    const handleSubmit = () => {
        if (!currentProject || !currentUser) return;
        setValidationMsg("");

        if (isEquipment) {
            const validRows = equipmentRows.filter((r) => r.equipmentType && r.shifts > 0);
            const emptyRows = equipmentRows.filter((r) => !r.equipmentType || r.shifts <= 0);
            if (emptyRows.length > 0 && validRows.length === 0) {
                setValidationMsg("请填写设备类型和台班数");
                return;
            }
            if (validRows.length === 0) {
                setValidationMsg("请至少添加一条设备记录");
                return;
            }
            const materials = validRows.map((r) => ({
                id: r.id,
                dailyRecordId: 0,
                requisitionNo: null,
                materialName: r.equipmentType,
                specification: r.specification,
                quantity: r.shifts,
                recordType: "设备" as string,
                equipmentType: r.equipmentType,
                equipmentShifts: r.shifts,
            }));
            addDailyRecord({
                projectId: currentProject.id,
                teamLeaderId: currentUser.id,
                date: new Date().toISOString().split("T")[0],
                moduleType: "材料",
                status: "待审核",
                reviewComment: null,
                reviewedAt: null,
                materials,
            });
        } else {
            const validRows = materialRows.filter((r) => r.materialName && r.quantity > 0);
            const emptyRows = materialRows.filter((r) => !r.materialName || r.quantity <= 0);
            if (emptyRows.length > 0 && validRows.length === 0) {
                setValidationMsg("请填写材料名称和数量");
                return;
            }
            if (validRows.length === 0) {
                setValidationMsg("请至少添加一条材料记录");
                return;
            }
            const materials = validRows.map((r) => ({
                id: r.id,
                dailyRecordId: 0,
                requisitionNo: r.requisitionNo || null,
                materialName: r.materialName,
                specification: r.specification,
                quantity: r.quantity,
                recordType: activeTab,
                equipmentType: null,
                equipmentShifts: null,
            }));
            addDailyRecord({
                projectId: currentProject.id,
                teamLeaderId: currentUser.id,
                date: new Date().toISOString().split("T")[0],
                moduleType: "材料",
                status: "待审核",
                reviewComment: null,
                reviewedAt: null,
                materials,
            });
        }
        router.push("/home");
    };

    const hasValidRows = isEquipment
        ? equipmentRows.some((r) => r.equipmentType && r.shifts > 0)
        : materialRows.some((r) => r.materialName && r.quantity > 0);

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
                        材料设备
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
            </div>

            {/* Tab Bar */}
            <div className="px-4 flex border-b border-eng-gray-100">
                {TABS.map((tab) => (
                    <button
                        key={tab.key}
                        onClick={() => { setActiveTab(tab.key); setValidationMsg(""); }}
                        className={`flex-1 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                            activeTab === tab.key
                                ? "border-eng-blue text-eng-blue"
                                : "border-transparent text-eng-gray-400"
                        }`}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Quick actions */}
            {!isEquipment && currentProject && (
                <div className="px-4 py-2 flex items-center gap-3">
                    <button
                        onClick={handleCopyYesterday}
                        className="text-xs text-eng-blue flex items-center gap-1"
                    >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                            <path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                        复制昨日
                    </button>
                    <MaterialQuickSelect onSelect={handleQuickSelect} />
                </div>
            )}

            {/* Content */}
            {currentProject ? (
                <div className="flex-1 px-4 py-2 space-y-3">
                    {isEquipment
                        ? equipmentRows.map((row) => (
                              <EquipmentRowCard
                                  key={row.id}
                                  row={row}
                                  onUpdate={(f, v) =>
                                      updateEquipmentRow(row.id, f, v)
                                  }
                                  onRemove={() => removeEquipmentRow(row.id)}
                              />
                          ))
                        : materialRows.map((row) => (
                              <MaterialRowCard
                                  key={row.id}
                                  row={row}
                                  activeTab={activeTab}
                                  onUpdate={(f, v) =>
                                      updateMaterialRow(row.id, f, v)
                                  }
                                  onRemove={() => removeMaterialRow(row.id)}
                              />
                          ))}

                    <button
                        onClick={isEquipment ? addEquipmentRow : addMaterialRow}
                        className="w-full h-10 border border-dashed border-eng-gray-300 rounded-card text-eng-blue text-sm"
                    >
                        + 添加一行
                    </button>
                </div>
            ) : (
                <div className="flex-1 flex items-center justify-center px-4">
                    <p className="text-sm text-gray-400">
                        {isTeamLeader ? "暂无负责的项目" : "请选择一个项目"}
                    </p>
                </div>
            )}

            {/* Summary + Submit */}
            <div className="px-4 py-3 bg-white border-t border-eng-gray-100 space-y-3">
                {validationMsg && (
                    <p className="text-xs text-red-500">{validationMsg}</p>
                )}
                <div className="flex items-center justify-between text-sm">
                    <span className="text-eng-gray-500">
                        共 {totalRows} 项
                    </span>
                    <span className="text-eng-gray-500">
                        {isEquipment ? "总台班" : "总数量"}：{totalQuantity}
                    </span>
                </div>
                <button
                    onClick={handleSubmit}
                    disabled={!hasValidRows || !currentProject}
                    className={`w-full h-11 rounded-btn text-sm font-medium transition-colors ${
                        hasValidRows && currentProject
                            ? "bg-eng-blue text-white hover:bg-blue-600"
                            : "bg-eng-gray-200 text-eng-gray-400 cursor-not-allowed"
                    }`}
                >
                    提交审核
                </button>
            </div>
        </div>
    );
}

function MaterialRowCard({
    row,
    activeTab,
    onUpdate,
    onRemove,
}: {
    row: MaterialRow;
    activeTab: string;
    onUpdate: (field: keyof MaterialRow, value: string | number) => void;
    onRemove: () => void;
}) {
    return (
        <div className="bg-white rounded-card border border-eng-gray-100 p-3 space-y-2">
            <div className="flex items-center justify-between">
                <span className="text-xs text-eng-gray-400">
                    {activeTab === "采买" ? "乙供材" : "材料"}
                </span>
                <button
                    onClick={onRemove}
                    className="text-eng-red text-xs"
                >
                    删除
                </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
                {activeTab !== "采买" && (
                    <input
                        type="text"
                        value={row.requisitionNo}
                        onChange={(e) =>
                            onUpdate("requisitionNo", e.target.value)
                        }
                        placeholder="领料单号（自动生成）"
                        className="h-9 px-2 border border-eng-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                    />
                )}
                <input
                    type="text"
                    value={row.materialName}
                    onChange={(e) =>
                        onUpdate("materialName", e.target.value)
                    }
                    placeholder="材料名称 *"
                    className="h-9 px-2 border border-eng-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                />
                <input
                    type="text"
                    value={row.specification}
                    onChange={(e) =>
                        onUpdate("specification", e.target.value)
                    }
                    placeholder="规格型号"
                    className="h-9 px-2 border border-eng-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                />
                <input
                    type="number"
                    min="0"
                    value={row.quantity || ""}
                    onChange={(e) =>
                        onUpdate("quantity", parseFloat(e.target.value) || 0)
                    }
                    placeholder="数量 *"
                    className="h-9 px-2 border border-eng-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                />
            </div>
        </div>
    );
}

function EquipmentRowCard({
    row,
    onUpdate,
    onRemove,
}: {
    row: EquipmentRow;
    onUpdate: (field: keyof EquipmentRow, value: string | number) => void;
    onRemove: () => void;
}) {
    return (
        <div className="bg-white rounded-card border border-eng-gray-100 p-3 space-y-2">
            <div className="flex items-center justify-between">
                <span className="text-xs text-eng-gray-400">设备</span>
                <button
                    onClick={onRemove}
                    className="text-eng-red text-xs"
                >
                    删除
                </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
                <select
                    value={row.equipmentType}
                    onChange={(e) => onUpdate("equipmentType", e.target.value)}
                    className="h-9 px-2 border border-eng-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue bg-white"
                >
                    <option value="">选择设备类型</option>
                    {EQUIPMENT_PRESETS.map((eq) => (
                        <option key={eq} value={eq}>{eq}</option>
                    ))}
                    <option value="custom">自定义...</option>
                </select>
                {row.equipmentType === "custom" ? (
                    <input
                        type="text"
                        onChange={(e) => onUpdate("equipmentType", e.target.value)}
                        placeholder="输入设备名称"
                        className="h-9 px-2 border border-eng-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                        autoFocus
                    />
                ) : (
                    <input
                        type="text"
                        value={row.specification}
                        onChange={(e) =>
                            onUpdate("specification", e.target.value)
                        }
                        placeholder="规格（如25吨）"
                        className="h-9 px-2 border border-eng-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                    />
                )}
                <input
                    type="number"
                    min="0"
                    value={row.shifts || ""}
                    onChange={(e) =>
                        onUpdate("shifts", parseFloat(e.target.value) || 0)
                    }
                    placeholder="台班数"
                    className="h-9 px-2 border border-eng-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                />
            </div>
        </div>
    );
}
