"use client";

import { useMockData } from "@/lib/MockDataProvider";
import { useRouter } from "next/navigation";

export default function ProfilePage() {
    const { currentUser, logout, getProjectsByTeamLeader, data } = useMockData();
    const router = useRouter();

    if (!currentUser) return null;

    const myProjects = getProjectsByTeamLeader(currentUser.id);
    const pendingCount = data.dailyRecords.filter(
        (r) =>
            myProjects.some((p) => p.id === r.projectId) &&
            r.status === "待审核"
    ).length;

    const handleLogout = () => {
        logout();
        router.push("/login");
    };

    return (
        <div className="px-4 py-4 space-y-4">
            {/* User Card */}
            <div className="bg-white rounded-card border border-eng-gray-100 p-4 flex items-center gap-4">
                <div className="w-14 h-14 bg-eng-blue rounded-full flex items-center justify-center text-white text-xl font-medium">
                    {currentUser.name[0]}
                </div>
                <div className="flex-1">
                    <h2 className="text-base font-bold text-eng-gray-900">
                        {currentUser.name}
                    </h2>
                    <p className="text-sm text-eng-gray-400">
                        {currentUser.phone}
                    </p>
                    <span className="inline-block text-xs px-2 py-0.5 bg-eng-blue/10 text-eng-blue rounded-full mt-1">
                        {currentUser.role}
                    </span>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-3">
                <div className="bg-white rounded-card border border-eng-gray-100 p-3 text-center">
                    <p className="text-xl font-bold text-eng-blue">
                        {myProjects.length}
                    </p>
                    <p className="text-xs text-eng-gray-400">负责项目</p>
                </div>
                <div className="bg-white rounded-card border border-eng-gray-100 p-3 text-center">
                    <p className="text-xl font-bold text-eng-orange">
                        {pendingCount}
                    </p>
                    <p className="text-xs text-eng-gray-400">待审核</p>
                </div>
                <div className="bg-white rounded-card border border-eng-gray-100 p-3 text-center">
                    <p className="text-xl font-bold text-eng-green">0</p>
                    <p className="text-xs text-eng-gray-400">离线草稿</p>
                </div>
            </div>

            {/* Menu Items */}
            <div className="bg-white rounded-card border border-eng-gray-100 divide-y divide-eng-gray-100">
                <MenuItem
                    label="负责项目"
                    value={`${myProjects.length} 个`}
                />
                <MenuItem
                    label="所属部门"
                    value="金卓南京项目部"
                />
                <MenuItem label="数据同步" value="已同步" />
            </div>

            <div className="bg-white rounded-card border border-eng-gray-100 divide-y divide-eng-gray-100">
                <MenuItem label="离线录入" value="" />
                <MenuItem label="关于工程管家" value="v1.0.0" />
            </div>

            {/* Logout */}
            <button
                onClick={handleLogout}
                className="w-full h-11 bg-white border border-eng-gray-200 rounded-btn text-eng-red text-sm font-medium hover:bg-red-50 transition-colors"
            >
                退出登录
            </button>
        </div>
    );
}

function MenuItem({ label, value }: { label: string; value: string }) {
    return (
        <div className="flex items-center justify-between px-4 py-3">
            <span className="text-sm text-eng-gray-700">{label}</span>
            <div className="flex items-center gap-1">
                <span className="text-sm text-eng-gray-400">{value}</span>
                <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={1.5}
                    className="text-eng-gray-300"
                >
                    <path d="M9 5l7 7-7 7" />
                </svg>
            </div>
        </div>
    );
}
