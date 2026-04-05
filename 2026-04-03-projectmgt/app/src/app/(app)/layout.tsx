"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useMockData } from "@/lib/MockDataProvider";
import { BottomTabBar } from "@/components/BottomTabBar";
import { AiChatWidget } from "@/components/ai/AiChatWidget";

function AuthGuard({ children }: { children: React.ReactNode }) {
    const { currentUser, hydrated } = useMockData();
    const pathname = usePathname();
    const router = useRouter();

    const hideTab = pathname === "/login";

    useEffect(() => {
        if (hydrated && !currentUser && pathname !== "/login") {
            router.replace("/login");
        }
    }, [hydrated, currentUser, pathname, router]);

    if (!hydrated) {
        return null;
    }

    if (!currentUser && !hideTab) {
        return null;
    }

    return (
        <div className="min-h-screen bg-eng-gray-50 pb-16">
            {children}
            {!hideTab && <BottomTabBar />}
            {!hideTab && currentUser?.role === "班组长" && <AiChatWidget />}
        </div>
    );
}

export default function AppLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <div className="max-w-app mx-auto bg-white min-h-screen relative shadow-lg">
            <AuthGuard>{children}</AuthGuard>
        </div>
    );
}
