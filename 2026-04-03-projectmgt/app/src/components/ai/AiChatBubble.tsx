"use client";

import React from "react";

interface AiChatBubbleProps {
    role: "user" | "assistant";
    content: string;
}

export function AiChatBubble({ role, content }: AiChatBubbleProps) {
    const isUser = role === "user";

    return (
        <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-2`}>
            <div
                className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                    isUser
                        ? "bg-eng-blue text-white rounded-br-none"
                        : "bg-gray-100 text-gray-800 rounded-bl-none"
                }`}
            >
                {content.split("**").map((part, i) =>
                    i % 2 === 1 ? (
                        <strong key={i}>{part}</strong>
                    ) : (
                        <span key={i}>{part}</span>
                    )
                )}
            </div>
        </div>
    );
}
