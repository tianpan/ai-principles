"use client";

import { useState, useCallback } from "react";

interface PhotoCaptureProps {
    photos: string[];
    onChange: (photos: string[]) => void;
    max?: number;
    label?: string;
}

export function PhotoCapture({
    photos,
    onChange,
    max = 3,
    label = "现场照片",
}: PhotoCaptureProps) {
    const [showSheet, setShowSheet] = useState(false);
    const [loading, setLoading] = useState(false);

    const handleCapture = useCallback(
        (mode: "camera" | "album") => {
            setShowSheet(false);
            setLoading(true);
            setTimeout(() => {
                const seed = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
                const width = mode === "camera" ? 400 : 600;
                const height = mode === "camera" ? 300 : 400;
                const url = `https://picsum.photos/seed/${seed}/${width}/${height}`;
                const newPhotos = [...photos, url];
                onChange(newPhotos);
                setLoading(false);
            }, 300);
        },
        [photos, onChange]
    );

    const handleRemove = useCallback(
        (index: number) => {
            const newPhotos = photos.filter((_, i) => i !== index);
            onChange(newPhotos);
        },
        [photos, onChange]
    );

    return (
        <div>
            <label className="block text-sm text-gray-600 mb-2">
                {label}
                <span className="text-gray-400 ml-1">
                    ({photos.length}/{max})
                </span>
            </label>

            <div className="flex gap-2 flex-wrap">
                {photos.map((url, i) => (
                    <div key={i} className="relative w-20 h-20 rounded-lg overflow-hidden border border-gray-200">
                        <img
                            src={url}
                            alt={`照片 ${i + 1}`}
                            className="w-full h-full object-cover"
                            loading="lazy"
                        />
                        <button
                            onClick={() => handleRemove(i)}
                            className="absolute top-0.5 right-0.5 w-5 h-5 bg-black/50 rounded-full flex items-center justify-center"
                        >
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth={2}>
                                <path d="M18 6L6 18M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                ))}

                {photos.length < max && (
                    <button
                        onClick={() => setShowSheet(true)}
                        disabled={loading}
                        className="w-20 h-20 rounded-lg border-2 border-dashed border-gray-300 flex flex-col items-center justify-center gap-1 text-gray-400 hover:border-eng-blue hover:text-eng-blue transition-colors disabled:opacity-50"
                    >
                        {loading ? (
                            <div className="w-5 h-5 border-2 border-eng-blue border-t-transparent rounded-full animate-spin" />
                        ) : (
                            <>
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                                    <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
                                    <circle cx="12" cy="13" r="4" />
                                </svg>
                                <span className="text-xs">添加</span>
                            </>
                        )}
                    </button>
                )}
            </div>

            {/* Bottom Sheet */}
            {showSheet && (
                <div className="fixed inset-0 z-50 flex items-end justify-center">
                    <div
                        className="absolute inset-0 bg-black/40"
                        onClick={() => setShowSheet(false)}
                    />
                    <div className="relative bg-white rounded-t-xl w-full max-w-app mx-auto safe-bottom">
                        <div className="px-4 pt-3 pb-1">
                            <div className="mx-auto w-10 h-1 bg-gray-300 rounded-full" />
                        </div>
                        <div className="p-4 space-y-2">
                            <button
                                onClick={() => handleCapture("camera")}
                                className="w-full py-3 text-sm text-eng-blue font-medium bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
                            >
                                拍照
                            </button>
                            <button
                                onClick={() => handleCapture("album")}
                                className="w-full py-3 text-sm text-gray-700 font-medium bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                            >
                                从相册选择
                            </button>
                            <button
                                onClick={() => setShowSheet(false)}
                                className="w-full py-3 text-sm text-gray-500 rounded-lg hover:bg-gray-50 transition-colors"
                            >
                                取消
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
