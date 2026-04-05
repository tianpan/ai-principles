"use client";

import React, { createContext, useContext, useState, useCallback, useMemo, useEffect } from "react";
import usersData from "../mock/users.json";
import projectsData from "../mock/projects.json";
import workersData from "../mock/workers.json";
import templatesData from "../mock/templates.json";
import dailyRecordsData from "../mock/daily-records.json";
import aiChatLogsData from "../mock/ai-chat-logs.json";
import aiAuditLogsData from "../mock/ai-audit-logs.json";
import aiBriefingsData from "../mock/ai-briefings.json";

// Type definitions
interface User {
    id: number;
    name: string;
    phone: string;
    password: string;
    role: string;
    departmentId: number | null;
    status: string;
    createdAt: string;
}

interface WorkItem {
    id: number;
    projectId: number;
    name: string;
    targetQuantity: number;
    unit: string;
    weight: number;
}

interface Project {
    id: number;
    name: string;
    departmentId: number;
    creatorId: number;
    startDate: string;
    endDate: string;
    teamLeaderId: number;
    memberIds: number[];
    remarks: string;
    status: string;
    isSubcontractor: boolean;
    constructionPlanUrls: string[];
    createdAt: string;
    workItems: WorkItem[];
}

interface Department {
    id: number;
    name: string;
    managerId: number;
    createdAt: string;
}

interface Worker {
    id: number;
    departmentId: number;
    name: string;
    phone: string;
    position: string;
    certifications: Array<{
        type: string;
        level: string;
        expireDate: string;
    }>;
}

interface Template {
    id: number;
    name: string;
    category: string;
    content: string;
    createdAt: string;
}

interface DailyRecord {
    id: number;
    projectId: number;
    teamLeaderId: number;
    date: string;
    moduleType: string;
    status: string;
    reviewComment: string | null;
    reviewedAt: string | null;
    attendance?: Record<string, unknown>;
    progress?: Array<Record<string, unknown>>;
    materials?: Array<Record<string, unknown>>;
    siteRecord?: Record<string, unknown>;
}

interface MockData {
    users: User[];
    departments: Department[];
    projects: Project[];
    workers: Worker[];
    templates: Template[];
    dailyRecords: DailyRecord[];
    aiChatLogs: AiChatLog[];
    aiAuditLogs: AiAuditLog[];
    aiBriefings: AiBriefing[];
}

interface AiChatLog {
    id: number;
    userId: number;
    projectId: number;
    sessionId: string;
    role: "user" | "assistant";
    content: string;
    parsedIntent?: {
        module: "考勤" | "进度" | "材料" | "施工记录";
        action: "create" | "query" | "update";
        fields: Record<string, unknown>;
    };
    createdAt: string;
}

interface AiAuditLog {
    id: number;
    dailyRecordId: number;
    result: "正常" | "疑似异常" | "明显问题";
    reason: string;
    confidence: number;
    humanDecision: "通过" | "退回" | null;
    createdAt: string;
}

interface AiBriefing {
    id: number;
    projectId: number;
    date: string;
    content: string;
    risks: Array<{
        type: string;
        severity: "高" | "中" | "低";
        description: string;
    }>;
    createdAt: string;
}

interface MockDataContextType {
    data: MockData;
    currentUser: User | null;
    hydrated: boolean;
    login: (phone: string, password: string) => User | null;
    logout: () => void;
    updateDailyRecord: (id: number, updates: Partial<DailyRecord>) => void;
    addDailyRecord: (record: Omit<DailyRecord, "id">) => DailyRecord;
    getProjectsByTeamLeader: (teamLeaderId: number) => Project[];
    getDailyRecordsByProject: (projectId: number) => DailyRecord[];
    getDailyRecordsByDate: (date: string) => DailyRecord[];
    getPendingRecords: () => DailyRecord[];
    getOverdueRecords: () => DailyRecord[];
    getWorkerById: (id: number) => Worker | undefined;
    getUserById: (id: number) => User | undefined;
    getProjectById: (id: number) => Project | undefined;
    getCumulativeProgress: (projectId: number, workItemId: number) => number;
    addAiChatLog: (log: Omit<AiChatLog, "id">) => AiChatLog;
    getAiChatLogsBySession: (sessionId: string) => AiChatLog[];
    addAiAuditLog: (log: Omit<AiAuditLog, "id">) => AiAuditLog;
    getAiAuditLogsByRecord: (dailyRecordId: number) => AiAuditLog | undefined;
    updateAiAuditLog: (id: number, updates: Partial<AiAuditLog>) => void;
    addAiBriefing: (briefing: Omit<AiBriefing, "id">) => AiBriefing;
    getAiBriefingByProjectAndDate: (projectId: number, date: string) => AiBriefing | undefined;
}

const MockDataContext = createContext<MockDataContextType | null>(null);

let nextRecordId = 100;

export function MockDataProvider({ children }: { children: React.ReactNode }) {
    const [data, setData] = useState<MockData>({
        users: usersData as User[],
        departments: projectsData.departments as Department[],
        projects: projectsData.projects as Project[],
        workers: workersData as Worker[],
        templates: templatesData as Template[],
        dailyRecords: dailyRecordsData.dailyRecords as DailyRecord[],
        aiChatLogs: aiChatLogsData as AiChatLog[],
        aiAuditLogs: aiAuditLogsData as AiAuditLog[],
        aiBriefings: aiBriefingsData as AiBriefing[],
    });
    const [currentUser, setCurrentUser] = useState<User | null>(null);
    const [hydrated, setHydrated] = useState(false);

    useEffect(() => {
        try {
            const saved = localStorage.getItem("mockCurrentUser");
            if (saved) {
                setCurrentUser(JSON.parse(saved));
            }
        } catch {}
        setHydrated(true);
    }, []);

    const login = useCallback(
        (phone: string, password: string): User | null => {
            const user = data.users.find(
                (u) => u.phone === phone && u.password === password && u.status === "启用"
            );
            if (user) {
                setCurrentUser(user);
                localStorage.setItem("mockCurrentUser", JSON.stringify(user));
            }
            return user || null;
        },
        [data.users]
    );

    const logout = useCallback(() => {
        setCurrentUser(null);
        localStorage.removeItem("mockCurrentUser");
    }, []);

    const updateDailyRecord = useCallback(
        (id: number, updates: Partial<DailyRecord>) => {
            setData((prev) => ({
                ...prev,
                dailyRecords: prev.dailyRecords.map((r) =>
                    r.id === id ? { ...r, ...updates } : r
                ),
            }));
        },
        []
    );

    const addDailyRecord = useCallback(
        (record: Omit<DailyRecord, "id">): DailyRecord => {
            const newRecord = { ...record, id: ++nextRecordId } as DailyRecord;
            setData((prev) => ({
                ...prev,
                dailyRecords: [...prev.dailyRecords, newRecord],
            }));
            return newRecord;
        },
        []
    );

    const getProjectsByTeamLeader = useCallback(
        (teamLeaderId: number) =>
            data.projects.filter((p) => p.teamLeaderId === teamLeaderId),
        [data.projects]
    );

    const getDailyRecordsByProject = useCallback(
        (projectId: number) =>
            data.dailyRecords.filter((r) => r.projectId === projectId),
        [data.dailyRecords]
    );

    const getDailyRecordsByDate = useCallback(
        (date: string) =>
            data.dailyRecords.filter((r) => r.date === date),
        [data.dailyRecords]
    );

    const getPendingRecords = useCallback(
        () => data.dailyRecords.filter((r) => r.status === "待审核"),
        [data.dailyRecords]
    );

    const getOverdueRecords = useCallback(() => {
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
        const cutoff = sevenDaysAgo.toISOString().split("T")[0];
        return data.dailyRecords.filter(
            (r) => r.status === "待审核" && r.date < cutoff
        );
    }, [data.dailyRecords]);

    const getWorkerById = useCallback(
        (id: number) => data.workers.find((w) => w.id === id),
        [data.workers]
    );

    const getUserById = useCallback(
        (id: number) => data.users.find((u) => u.id === id),
        [data.users]
    );

    const getProjectById = useCallback(
        (id: number) => data.projects.find((p) => p.id === id),
        [data.projects]
    );

    const getCumulativeProgress = useCallback(
        (projectId: number, workItemId: number): number => {
            return data.dailyRecords
                .filter(
                    (r) =>
                        r.projectId === projectId &&
                        r.moduleType === "进度" &&
                        r.status !== "已退回"
                )
                .reduce((sum, r) => {
                    const progress = r.progress as
                        | Array<{ workItemId: number; completedQuantity: number }>
                        | undefined;
                    const item = progress?.find(
                        (p) => p.workItemId === workItemId
                    );
                    return sum + (item?.completedQuantity || 0);
                }, 0);
        },
        [data.dailyRecords]
    );

    const addAiChatLog = useCallback(
        (log: Omit<AiChatLog, "id">): AiChatLog => {
            const newLog = { ...log, id: ++nextRecordId } as AiChatLog;
            setData((prev) => ({
                ...prev,
                aiChatLogs: [...prev.aiChatLogs, newLog],
            }));
            return newLog;
        },
        []
    );

    const getAiChatLogsBySession = useCallback(
        (sessionId: string) =>
            data.aiChatLogs.filter((l) => l.sessionId === sessionId),
        [data.aiChatLogs]
    );

    const addAiAuditLog = useCallback(
        (log: Omit<AiAuditLog, "id">): AiAuditLog => {
            const newLog = { ...log, id: ++nextRecordId } as AiAuditLog;
            setData((prev) => ({
                ...prev,
                aiAuditLogs: [...prev.aiAuditLogs, newLog],
            }));
            return newLog;
        },
        []
    );

    const getAiAuditLogsByRecord = useCallback(
        (dailyRecordId: number) =>
            data.aiAuditLogs.find((l) => l.dailyRecordId === dailyRecordId),
        [data.aiAuditLogs]
    );

    const updateAiAuditLog = useCallback(
        (id: number, updates: Partial<AiAuditLog>) => {
            setData((prev) => ({
                ...prev,
                aiAuditLogs: prev.aiAuditLogs.map((l) =>
                    l.id === id ? { ...l, ...updates } : l
                ),
            }));
        },
        []
    );

    const addAiBriefing = useCallback(
        (briefing: Omit<AiBriefing, "id">): AiBriefing => {
            const newBriefing = { ...briefing, id: ++nextRecordId } as AiBriefing;
            setData((prev) => ({
                ...prev,
                aiBriefings: [...prev.aiBriefings, newBriefing],
            }));
            return newBriefing;
        },
        []
    );

    const getAiBriefingByProjectAndDate = useCallback(
        (projectId: number, date: string) =>
            data.aiBriefings.find(
                (b) => b.projectId === projectId && b.date === date
            ),
        [data.aiBriefings]
    );

    const value = useMemo(
        () => ({
            data,
            currentUser,
            hydrated,
            login,
            logout,
            updateDailyRecord,
            addDailyRecord,
            getProjectsByTeamLeader,
            getDailyRecordsByProject,
            getDailyRecordsByDate,
            getPendingRecords,
            getOverdueRecords,
            getWorkerById,
            getUserById,
            getProjectById,
            getCumulativeProgress,
            addAiChatLog,
            getAiChatLogsBySession,
            addAiAuditLog,
            getAiAuditLogsByRecord,
            updateAiAuditLog,
            addAiBriefing,
            getAiBriefingByProjectAndDate,
        }),
        [
            data, currentUser, hydrated, login, logout, updateDailyRecord, addDailyRecord,
            getProjectsByTeamLeader, getDailyRecordsByProject, getDailyRecordsByDate,
            getPendingRecords, getOverdueRecords, getWorkerById, getUserById,
            getProjectById, getCumulativeProgress,
            addAiChatLog, getAiChatLogsBySession,
            addAiAuditLog, getAiAuditLogsByRecord, updateAiAuditLog,
            addAiBriefing, getAiBriefingByProjectAndDate,
        ]
    );

    return React.createElement(MockDataContext.Provider, { value }, children);
}

export function useMockData(): MockDataContextType {
    const context = useContext(MockDataContext);
    if (!context) {
        throw new Error("useMockData must be used within MockDataProvider");
    }
    return context;
}
