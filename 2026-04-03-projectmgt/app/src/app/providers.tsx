"use client";

import { MockDataProvider } from "@/lib/MockDataProvider";

export default function Providers({ children }: { children: React.ReactNode }) {
    return <MockDataProvider>{children}</MockDataProvider>;
}
