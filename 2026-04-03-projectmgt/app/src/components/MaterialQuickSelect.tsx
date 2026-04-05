"use client";

import { useState } from "react";

const COMMON_MATERIALS = [
    { name: "PE管 DN110", unit: "米" },
    { name: "PE管 DN160", unit: "米" },
    { name: "PE管 DN200", unit: "米" },
    { name: "钢管 DN100", unit: "米" },
    { name: "钢管 DN150", unit: "米" },
    { name: "阀门 DN100", unit: "个" },
    { name: "阀门 DN150", unit: "个" },
    { name: "弯头 DN110", unit: "个" },
    { name: "三通 DN110", unit: "个" },
    { name: "法兰 DN100", unit: "片" },
    { name: "调压箱", unit: "台" },
    { name: "警示带", unit: "米" },
    { name: "燃气表 G4", unit: "台" },
    { name: "铝塑管 DN20", unit: "米" },
    { name: "球阀 DN20", unit: "个" },
];

interface MaterialItem {
    name: string;
    unit: string;
    quantity: number;
    remark: string;
}

interface MaterialQuickSelectProps {
    onSelect: (item: MaterialItem) => void;
}

export function MaterialQuickSelect({ onSelect }: MaterialQuickSelectProps) {
    const [show, setShow] = useState(false);
    const [search, setSearch] = useState("");

    const filtered = COMMON_MATERIALS.filter((m) =>
        m.name.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div>
            <button
                type="button"
                onClick={() => setShow(!show)}
                className="text-xs text-eng-blue hover:underline"
            >
                {show ? "收起常用材料" : "快捷选择常用材料"}
            </button>

            {show && (
                <div className="mt-2 border border-gray-200 rounded-lg p-3 bg-gray-50">
                    <input
                        type="text"
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        placeholder="搜索材料名称"
                        className="w-full h-8 px-3 border border-gray-200 rounded text-xs mb-2 focus:outline-none focus:border-eng-blue"
                    />
                    <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto">
                        {filtered.map((m) => (
                            <button
                                key={m.name}
                                type="button"
                                onClick={() =>
                                    onSelect({
                                        name: m.name,
                                        unit: m.unit,
                                        quantity: 1,
                                        remark: "",
                                    })
                                }
                                className="px-2 py-1 text-xs bg-white border border-gray-200 rounded hover:border-eng-blue hover:text-eng-blue transition-colors"
                            >
                                {m.name}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export function generateMaterialNo(): string {
    const now = new Date();
    const y = now.getFullYear();
    const m = String(now.getMonth() + 1).padStart(2, "0");
    const d = String(now.getDate()).padStart(2, "0");
    const seq = String(Math.floor(Math.random() * 900) + 100);
    return `MR-${y}${m}${d}-${seq}`;
}
