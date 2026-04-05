"use client";

import { useMockData } from "@/lib/MockDataProvider";

interface ProjectSelectorProps {
    value: number | null;
    onChange: (projectId: number) => void;
    className?: string;
}

export function ProjectSelector({
    value,
    onChange,
    className = "",
}: ProjectSelectorProps) {
    const { data } = useMockData();

    return (
        <select
            value={value ?? ""}
            onChange={(e) => onChange(Number(e.target.value))}
            className={`h-9 px-3 border border-gray-200 rounded text-sm text-gray-700 focus:outline-none focus:border-eng-blue bg-white ${className}`}
        >
            <option value="" disabled>
                选择项目
            </option>
            {data.projects.map((p) => (
                <option key={p.id} value={p.id}>
                    {p.name}
                </option>
            ))}
        </select>
    );
}
