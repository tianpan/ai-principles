/**
 * LearningProgress - 学习进度指示器
 *
 * 显示当前学习进度：已完成/总页数
 * 用于侧边栏或页面顶部
 */

import { useState, useEffect } from 'react';

interface LearningProgressProps {
  /** 总页数 */
  totalPages?: number;
  /** 存储键名 */
  storageKey?: string;
}

// 学习路径定义
const LEARNING_PATH = [
  { id: 'prerequisites', label: '预备知识', section: '基础' },
  { id: 'chapter2/1-token-embedding', label: 'Token & Embedding', section: 'Chapter 2' },
  { id: 'chapter2/2-self-attention', label: 'Self-Attention', section: 'Chapter 2' },
  { id: 'chapter2/3-multi-head-attention', label: 'Multi-Head', section: 'Chapter 2' },
  { id: 'chapter2/4-positional-encoding', label: '位置编码', section: 'Chapter 2' },
  { id: 'chapter2/5-residual-layernorm', label: '残差+LayerNorm', section: 'Chapter 2' },
  { id: 'chapter2/6-ffn-output', label: 'FFN & 输出', section: 'Chapter 2' },
  { id: 'lab/intro', label: '实验室介绍', section: '实践' },
  { id: 'lab/implementation', label: '代码实现', section: '实践' },
  { id: 'lab/training', label: '训练与推理', section: '实践' },
];

export default function LearningProgress({
  totalPages = LEARNING_PATH.length,
  storageKey = 'ai-principles-progress'
}: LearningProgressProps) {
  const [visitedPages, setVisitedPages] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState<string>('');

  useEffect(() => {
    // 从 localStorage 加载已访问的页面
    const stored = localStorage.getItem(storageKey);
    if (stored) {
      setVisitedPages(new Set(JSON.parse(stored)));
    }

    // 获取当前页面
    const path = window.location.pathname.replace(/^\//, '').replace(/\/$/, '');
    setCurrentPage(path);

    // 标记当前页面为已访问
    if (path) {
      const newVisited = new Set(visitedPages);
      newVisited.add(path);
      setVisitedPages(newVisited);
      localStorage.setItem(storageKey, JSON.stringify([...newVisited]));
    }
  }, [storageKey]);

  const completedCount = visitedPages.size;
  const progress = Math.round((completedCount / totalPages) * 100);
  const currentIndex = LEARNING_PATH.findIndex(p => p.id === currentPage);

  return (
    <div className="learning-progress">
      {/* 进度条 */}
      <div className="mb-3">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-slate-400">学习进度</span>
          <span className="text-cyan-400 font-medium">{progress}%</span>
        </div>
        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 to-green-500 transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* 步骤指示 */}
      <div className="flex items-center gap-2 text-xs text-slate-500">
        <span>{completedCount}/{totalPages} 页</span>
        {currentIndex >= 0 && (
          <>
            <span>•</span>
            <span>当前位置: {LEARNING_PATH[currentIndex]?.label}</span>
          </>
        )}
      </div>

      {/* 下一步提示 */}
      {currentIndex < LEARNING_PATH.length - 1 && (
        <div className="mt-3 p-2 bg-slate-800/50 rounded-lg text-xs">
          <span className="text-slate-400">下一步: </span>
          <span className="text-cyan-400">{LEARNING_PATH[currentIndex + 1]?.label}</span>
        </div>
      )}

      {progress === 100 && (
        <div className="mt-3 p-2 bg-green-900/30 rounded-lg text-xs text-green-400">
          🎉 恭喜完成所有章节！
        </div>
      )}
    </div>
  );
}
