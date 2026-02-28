import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function MultiHeadComparator() {
  const [numHeads, setNumHeads] = useState(4);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const tokens = ['The', 'cat', 'sat', 'on', 'mat'];

  // Simulate multi-head attention patterns
  const headPatterns = useMemo(() => {
    const patterns: { [key: number]: number[][][] } = {};

    for (let h = 1; h <= 8; h++) {
      patterns[h] = [];
      const dimPerHead = Math.floor(64 / h);

      for (let headIdx = 0; headIdx < h; headIdx++) {
        const weights: number[][] = [];
        const phase = (headIdx / h) * Math.PI * 2;
        const frequency = 0.5 + (headIdx % 3) * 0.3;

        for (let i = 0; i < tokens.length; i++) {
          const row: number[] = [];
          let sum = 0;

          for (let j = 0; j < tokens.length; j++) {
            const dist = Math.abs(i - j);
            const pattern = Math.sin(dist * frequency + phase) * 0.5 + 0.5;
            const local = j >= Math.max(0, i - 2) && j <= i ? 0.3 : 0;
            const base = pattern * 0.7 + local;
            row.push(base);
            sum += base;
          }

          weights.push(row.map(w => w / sum));
        }

        patterns[h].push(weights);
      }
    }

    return patterns;
  }, [tokens]);

  // Get merged attention (average of all heads)
  const getMergedAttention = (numH: number): number[][] => {
    const heads = headPatterns[numH];
    const merged: number[][] = [];

    for (let i = 0; i < tokens.length; i++) {
      merged.push([]);
      for (let j = 0; j < tokens.length; j++) {
        const avg = heads.reduce((sum, h) => sum + h[i][j], 0) / heads.length;
        merged[i].push(avg);
      }
    }

    return merged;
  };

  const getColor = (weight: number) => {
    const intensity = Math.min(weight * 3, 1);
    return `rgba(6, 182, 212, ${intensity})`;
  };

  const handleHeadChange = (newVal: number) => {
    setIsTransitioning(true);
    setNumHeads(newVal);
    setTimeout(() => setIsTransitioning(false), 300);
  };

  const renderAttentionMatrix = (weights: number[][], title: string, compact = false, headIndex = -1) => (
    <motion.div
      key={`${title}-${headIndex}`}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.3 }}
      className="text-center"
    >
      <div className="text-xs text-slate-400 mb-1">{title}</div>
      <div className="inline-block rounded overflow-hidden">
        {weights.map((row, i) => (
          <div key={i} className="flex">
            {row.map((w, j) => (
              <motion.div
                key={j}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: (i * weights.length + j) * 0.02, type: "spring", stiffness: 300 }}
                className={`${compact ? 'w-6 h-6' : 'w-8 h-8'} flex items-center justify-center text-xs`}
                style={{ backgroundColor: getColor(w) }}
              >
                {compact ? '' : w.toFixed(1)}
              </motion.div>
            ))}
          </div>
        ))}
      </div>
    </motion.div>
  );

  return (
    <div className="visualization-container">
      <h3 className="text-lg font-bold text-cyan-400 mb-4">Multi-Head Attention 对比器</h3>

      {/* Head selector with animation */}
      <div className="mb-6">
        <label className="block text-slate-400 mb-2">
          注意力头数量: <span className="text-cyan-400 font-bold text-xl">{numHeads}</span>
        </label>
        <div className="relative">
          <input
            type="range"
            min={1}
            max={8}
            value={numHeads}
            onChange={(e) => handleHeadChange(parseInt(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />
          {/* Animated head indicators */}
          <div className="flex justify-between mt-2">
            {[1, 2, 3, 4, 5, 6, 7, 8].map((h) => (
              <motion.div
                key={h}
                initial={{ scale: 0 }}
                animate={{
                  scale: h <= numHeads ? 1 : 0.6,
                  backgroundColor: h <= numHeads ? '#06b6d4' : '#334155'
                }}
                className="w-6 h-6 rounded-full flex items-center justify-center text-xs text-white font-bold"
              >
                {h}
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Single head vs Multi-head comparison */}
      <AnimatePresence mode="wait">
        <motion.div
          key={numHeads}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          className="grid grid-cols-2 gap-6 mb-6"
        >
          <div className="bg-slate-800/50 p-4 rounded-lg border border-amber-500/20">
            <h4 className="text-center font-bold text-amber-400 mb-4">单头 Attention</h4>
            {renderAttentionMatrix(headPatterns[1][0], '1 个头')}
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="text-xs text-slate-500 mt-3 text-center"
            >
              只能学习一种注意力模式
            </motion.p>
          </div>

          <div className="bg-slate-800/50 p-4 rounded-lg border border-green-500/20">
            <h4 className="text-center font-bold text-green-400 mb-4">
              {numHeads} 头 Attention (合并后)
            </h4>
            {renderAttentionMatrix(getMergedAttention(numHeads), `${numHeads} 个头平均`)}
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="text-xs text-slate-500 mt-3 text-center"
            >
              多个头的模式融合在一起
            </motion.p>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Individual heads with staggered animation */}
      {numHeads > 1 && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mb-6 overflow-hidden"
        >
          <h4 className="text-sm font-bold text-slate-400 mb-3">各个头的独立模式:</h4>
          <div className="grid grid-cols-4 gap-2">
            <AnimatePresence>
              {headPatterns[numHeads].slice(0, numHeads).map((weights, idx) => (
                <motion.div
                  key={`${numHeads}-${idx}`}
                  initial={{ opacity: 0, scale: 0, rotateY: 90 }}
                  animate={{ opacity: 1, scale: 1, rotateY: 0 }}
                  exit={{ opacity: 0, scale: 0, rotateY: -90 }}
                  transition={{
                    delay: idx * 0.1,
                    type: "spring",
                    stiffness: 200,
                    damping: 15
                  }}
                  className="bg-slate-800/30 p-2 rounded border border-cyan-500/10"
                >
                  {renderAttentionMatrix(weights, `Head ${idx + 1}`, true, idx)}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </motion.div>
      )}

      {/* Explanation with animated icons */}
      <div className="bg-slate-800/50 p-4 rounded-lg border border-cyan-500/20">
        <h4 className="font-bold text-cyan-400 mb-2">观察要点</h4>
        <ul className="text-sm text-slate-300 space-y-2">
          {[
            { icon: '1️⃣', text: '单头：只能捕获一种注意力模式' },
            { icon: '🎯', text: '多头：每个头可以学习不同的模式（局部、全局、语法关系等）' },
            { icon: '✨', text: '合并后：多个模式融合，表达能力更强' },
          ].map((item, i) => (
            <motion.li
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex items-center gap-2"
            >
              <span>{item.icon}</span>
              <span>{item.text}</span>
            </motion.li>
          ))}
        </ul>
      </div>
    </div>
  );
}
