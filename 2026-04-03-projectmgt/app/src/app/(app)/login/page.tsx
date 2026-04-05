"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMockData } from "@/lib/MockDataProvider";

export default function LoginPage() {
    const router = useRouter();
    const { login, data } = useMockData();
    const [phone, setPhone] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");

    const handleLogin = () => {
        setError("");
        const user = login(phone, password);
        if (user) {
            router.push("/home");
        } else {
            setError("手机号或密码错误，或账号已停用");
        }
    };

    return (
        <div className="min-h-screen flex flex-col items-center justify-center px-8 bg-white">
            {/* Logo Area */}
            <div className="mb-10 text-center">
                <div className="w-20 h-20 bg-eng-blue rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <svg
                        width="40"
                        height="40"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="white"
                        strokeWidth={1.5}
                    >
                        <path d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                    </svg>
                </div>
                <h1 className="text-xl font-bold text-eng-gray-900">工程管家</h1>
                <p className="text-sm text-eng-gray-400 mt-1">
                    建筑工程项目部精细化管理
                </p>
            </div>

            {/* Form */}
            <div className="w-full max-w-sm space-y-4">
                <div>
                    <label className="block text-sm text-eng-gray-600 mb-1">
                        手机号
                    </label>
                    <input
                        type="tel"
                        value={phone}
                        onChange={(e) => setPhone(e.target.value)}
                        placeholder="请输入手机号"
                        className="w-full h-11 px-3 border border-eng-gray-200 rounded-btn text-sm focus:outline-none focus:border-eng-blue"
                    />
                </div>

                <div>
                    <label className="block text-sm text-eng-gray-600 mb-1">
                        密码
                    </label>
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="请输入密码"
                        className="w-full h-11 px-3 border border-eng-gray-200 rounded-btn text-sm focus:outline-none focus:border-eng-blue"
                    />
                </div>

                {error && (
                    <p className="text-eng-red text-xs">{error}</p>
                )}

                <button
                    onClick={handleLogin}
                    className="w-full h-11 bg-eng-blue text-white rounded-btn text-sm font-medium hover:bg-blue-600 transition-colors"
                >
                    登录
                </button>

                {/* Quick Login Hints */}
                <div className="pt-4 border-t border-eng-gray-100 mt-6">
                    <p className="text-xs text-eng-gray-400 mb-2">
                        测试账号：
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                        {data.users
                            .filter((u) => u.status === "启用")
                            .map((u) => (
                                <button
                                    key={u.id}
                                    onClick={() => {
                                        setPhone(u.phone);
                                        setPassword(u.password);
                                    }}
                                    className="text-xs px-2 py-1.5 bg-eng-gray-50 rounded text-eng-gray-600 hover:bg-eng-gray-100 transition-colors"
                                >
                                    {u.name} ({u.role})
                                </button>
                            ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
