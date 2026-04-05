"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMockData } from "@/lib/MockDataProvider";

interface WorkItemInput {
    name: string;
    targetQuantity: number;
    unit: string;
    weight: number;
}

export default function PcNewProjectPage() {
    const router = useRouter();
    const { currentUser, data } = useMockData();

    const [name, setName] = useState("");
    const [departmentId, setDepartmentId] = useState(
        data.departments[0]?.id || 1
    );
    const [teamLeaderId, setTeamLeaderId] = useState(2);
    const [startDate, setStartDate] = useState("");
    const [endDate, setEndDate] = useState("");
    const [remarks, setRemarks] = useState("");
    const [isSubcontractor, setIsSubcontractor] = useState(false);
    const [workItems, setWorkItems] = useState<WorkItemInput[]>([
        { name: "", targetQuantity: 0, unit: "", weight: 0 },
    ]);

    if (!currentUser) return null;

    const teamLeaders = data.users.filter(
        (u) => u.role === "班组长" && u.status === "启用"
    );

    const weightSum = workItems.reduce((s, w) => s + w.weight, 0);
    const weightValid = Math.abs(weightSum - 100) < 0.01;

    const addWorkItem = () => {
        setWorkItems((prev) => [
            ...prev,
            { name: "", targetQuantity: 0, unit: "", weight: 0 },
        ]);
    };

    const removeWorkItem = (index: number) => {
        setWorkItems((prev) => prev.filter((_, i) => i !== index));
    };

    const updateWorkItem = (
        index: number,
        field: keyof WorkItemInput,
        value: string | number
    ) => {
        setWorkItems((prev) =>
            prev.map((item, i) =>
                i === index ? { ...item, [field]: value } : item
            )
        );
    };

    const handleSubmit = () => {
        // Mock submit - just go back
        alert(
            "项目创建成功（Mock模式，数据未持久化）\n" +
                JSON.stringify({ name, departmentId, teamLeaderId, startDate, endDate, workItems }, null, 2)
        );
        router.push("/pc/projects");
    };

    const isValid =
        name.trim() &&
        startDate &&
        endDate &&
        workItems.every((w) => w.name && w.unit && w.targetQuantity > 0) &&
        weightValid;

    return (
        <div className="max-w-3xl space-y-6">
            <div className="flex items-center gap-2">
                <button
                    onClick={() => router.back()}
                    className="text-gray-500"
                >
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                        <path d="M15 19l-7-7 7-7" />
                    </svg>
                </button>
                <h1 className="text-xl font-bold text-gray-800">
                    新建项目
                </h1>
            </div>

            {/* Basic Info */}
            <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-4">
                <h2 className="text-base font-medium text-gray-700">
                    基本信息
                </h2>

                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm text-gray-600 mb-1">
                            项目名称 *
                        </label>
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            className="w-full h-9 px-3 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-gray-600 mb-1">
                            所属部门
                        </label>
                        <select
                            value={departmentId}
                            onChange={(e) =>
                                setDepartmentId(Number(e.target.value))
                            }
                            className="w-full h-9 px-3 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                        >
                            {data.departments.map((d) => (
                                <option key={d.id} value={d.id}>
                                    {d.name}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-gray-600 mb-1">
                            班组长 *
                        </label>
                        <select
                            value={teamLeaderId}
                            onChange={(e) =>
                                setTeamLeaderId(Number(e.target.value))
                            }
                            className="w-full h-9 px-3 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                        >
                            {teamLeaders.map((u) => (
                                <option key={u.id} value={u.id}>
                                    {u.name}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="flex items-end">
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={isSubcontractor}
                                onChange={(e) =>
                                    setIsSubcontractor(e.target.checked)
                                }
                                className="accent-eng-blue"
                            />
                            <span className="text-sm text-gray-600">
                                分包项目
                            </span>
                        </label>
                    </div>
                    <div>
                        <label className="block text-sm text-gray-600 mb-1">
                            开始日期 *
                        </label>
                        <input
                            type="date"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            className="w-full h-9 px-3 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-gray-600 mb-1">
                            结束日期 *
                        </label>
                        <input
                            type="date"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            className="w-full h-9 px-3 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                        />
                    </div>
                </div>

                <div>
                    <label className="block text-sm text-gray-600 mb-1">
                        备注
                    </label>
                    <textarea
                        value={remarks}
                        onChange={(e) => setRemarks(e.target.value)}
                        className="w-full h-20 px-3 py-2 border border-gray-200 rounded text-sm resize-none focus:outline-none focus:border-eng-blue"
                    />
                </div>
            </div>

            {/* Work Items */}
            <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-4">
                <div className="flex items-center justify-between">
                    <h2 className="text-base font-medium text-gray-700">
                        工序清单
                    </h2>
                    <span
                        className={`text-sm ${
                            weightValid
                                ? "text-eng-green"
                                : "text-eng-red"
                        }`}
                    >
                        权重合计: {weightSum.toFixed(0)}%
                        {!weightValid && " (必须等于100%)"}
                    </span>
                </div>

                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-gray-100">
                            <th className="text-left py-2 text-gray-500 font-medium">
                                工序名称
                            </th>
                            <th className="text-left py-2 text-gray-500 font-medium">
                                目标量
                            </th>
                            <th className="text-left py-2 text-gray-500 font-medium">
                                单位
                            </th>
                            <th className="text-left py-2 text-gray-500 font-medium">
                                权重(%)
                            </th>
                            <th className="w-16"></th>
                        </tr>
                    </thead>
                    <tbody>
                        {workItems.map((item, i) => (
                            <tr key={i} className="border-b border-gray-50">
                                <td className="py-2 pr-2">
                                    <input
                                        type="text"
                                        value={item.name}
                                        onChange={(e) =>
                                            updateWorkItem(i, "name", e.target.value)
                                        }
                                        className="w-full h-8 px-2 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                                    />
                                </td>
                                <td className="py-2 pr-2">
                                    <input
                                        type="number"
                                        min="0"
                                        value={item.targetQuantity || ""}
                                        onChange={(e) =>
                                            updateWorkItem(
                                                i,
                                                "targetQuantity",
                                                parseFloat(e.target.value) || 0
                                            )
                                        }
                                        className="w-24 h-8 px-2 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                                    />
                                </td>
                                <td className="py-2 pr-2">
                                    <input
                                        type="text"
                                        value={item.unit}
                                        onChange={(e) =>
                                            updateWorkItem(i, "unit", e.target.value)
                                        }
                                        className="w-20 h-8 px-2 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                                    />
                                </td>
                                <td className="py-2 pr-2">
                                    <input
                                        type="number"
                                        min="0"
                                        max="100"
                                        value={item.weight || ""}
                                        onChange={(e) =>
                                            updateWorkItem(
                                                i,
                                                "weight",
                                                parseFloat(e.target.value) || 0
                                            )
                                        }
                                        className="w-20 h-8 px-2 border border-gray-200 rounded text-sm focus:outline-none focus:border-eng-blue"
                                    />
                                </td>
                                <td className="py-2">
                                    {workItems.length > 1 && (
                                        <button
                                            onClick={() => removeWorkItem(i)}
                                            className="text-eng-red text-xs"
                                        >
                                            删除
                                        </button>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>

                <button
                    onClick={addWorkItem}
                    className="text-eng-blue text-sm flex items-center gap-1"
                >
                    + 添加工序
                </button>
            </div>

            {/* Submit */}
            <div className="flex justify-end gap-3">
                <button
                    onClick={() => router.back()}
                    className="h-10 px-6 border border-gray-200 rounded text-sm text-gray-600"
                >
                    取消
                </button>
                <button
                    onClick={handleSubmit}
                    disabled={!isValid}
                    className={`h-10 px-6 rounded text-sm font-medium ${
                        isValid
                            ? "bg-eng-blue text-white hover:bg-blue-600"
                            : "bg-gray-200 text-gray-400 cursor-not-allowed"
                    }`}
                >
                    创建项目
                </button>
            </div>
        </div>
    );
}
