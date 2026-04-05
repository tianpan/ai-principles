import type { Metadata } from "next";
import "./globals.css";
import Providers from "./providers";

export const metadata: Metadata = {
    title: "工程管家",
    description: "建筑工程项目部精细化管理工具",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="zh-CN">
            <body className="antialiased">
                <Providers>{children}</Providers>
            </body>
        </html>
    );
}
