"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const tabs = [
    { href: "/home", label: "首页", icon: HomeIcon },
    { href: "/history", label: "历史", icon: HistoryIcon },
    { href: "/review", label: "审核", icon: ReviewIcon },
    { href: "/profile", label: "我的", icon: ProfileIcon },
];

export function BottomTabBar() {
    const pathname = usePathname();

    return (
        <nav className="fixed bottom-0 left-1/2 -translate-x-1/2 w-full max-w-app bg-white border-t border-eng-gray-200 z-50">
            <div className="flex justify-around items-center h-14">
                {tabs.map((tab) => {
                    const isActive =
                        pathname === tab.href ||
                        pathname.startsWith(tab.href + "/");
                    return (
                        <Link
                            key={tab.href}
                            href={tab.href}
                            className={`flex flex-col items-center justify-center gap-0.5 flex-1 h-full ${
                                isActive
                                    ? "text-eng-blue"
                                    : "text-eng-gray-400"
                            }`}
                        >
                            <tab.icon active={isActive} />
                            <span className="text-xs">{tab.label}</span>
                        </Link>
                    );
                })}
            </div>
        </nav>
    );
}

function HomeIcon({ active }: { active: boolean }) {
    return (
        <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill={active ? "currentColor" : "none"}
            stroke="currentColor"
            strokeWidth={active ? 0 : 1.5}
        >
            <path d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1" />
        </svg>
    );
}

function HistoryIcon({ active }: { active: boolean }) {
    return (
        <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={active ? 2 : 1.5}
        >
            <path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
    );
}

function ReviewIcon({ active }: { active: boolean }) {
    return (
        <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill={active ? "currentColor" : "none"}
            stroke="currentColor"
            strokeWidth={active ? 0 : 1.5}
        >
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
        </svg>
    );
}

function ProfileIcon({ active }: { active: boolean }) {
    return (
        <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill={active ? "currentColor" : "none"}
            stroke="currentColor"
            strokeWidth={active ? 0 : 1.5}
        >
            <path d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
    );
}
