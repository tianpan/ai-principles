// src/components/hero/AttentionViz.tsx
import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';

const tokens = ['The', 'cat', 'sat', 'on', 'mat'];

// 模拟注意力权重
const generateAttention = () => {
  return tokens.map((_, i) =>
    tokens.map((_, j) => {
      if (j > i) return 0; // 因果掩码
      const dist = i - j;
      return Math.exp(-dist * 0.3) + Math.random() * 0.2;
    }).map((v, _, arr) => v / arr.reduce((a, b) => a + b, 0))
  );
};

interface AttentionVizProps {
  isActive?: boolean;
}

export default function AttentionViz({ isActive = true }: AttentionVizProps) {
  const [attention, setAttention] = useState<number[][]>([]);

  useEffect(() => {
    setAttention(generateAttention());
    if (!isActive) return;

    const interval = setInterval(() => {
      setAttention(generateAttention());
    }, 3000);
    return () => clearInterval(interval);
  }, [isActive]);

  const getColor = (weight: number) => {
    if (weight === 0) return 'bg-slate-800';
    const intensity = Math.min(weight * 4, 1);
    return `rgba(6, 182, 212, ${intensity})`;
  };

  return (
    <div className="my-8">
      <motion.h3
        className="text-center font-bold mb-4"
        animate={{
          color: isActive ? '#22d3ee' : '#64748b',
        }}
      >
        Attention 热力图
      </motion.h3>

      <div className="flex justify-center">
        <div>
          {/* 列标题 */}
          <div className="flex mb-2">
            <div className="w-12 h-8" />
            {tokens.map((t, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.1 }}
                className="w-16 h-8 flex items-center justify-center text-slate-400 text-sm"
              >
                {t}
              </motion.div>
            ))}
          </div>

          {/* 热力图 */}
          {attention.map((row, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex"
            >
              <div className="w-12 h-12 flex items-center justify-center text-slate-400 text-sm">
                {tokens[i]}
              </div>
              {row.map((w, j) => (
                <motion.div
                  key={j}
                  initial={{ scale: 0 }}
                  animate={{
                    scale: isActive ? 1 : 0.95,
                  }}
                  whileHover={isActive ? { scale: 1.1 } : undefined}
                  transition={{ delay: i * 0.1 + j * 0.05 }}
                  className="w-16 h-12 flex items-center justify-center text-xs font-mono rounded m-0.5 transition-colors"
                  style={{
                    backgroundColor: getColor(w),
                    opacity: isActive ? 1 : 0.5,
                  }}
                >
                  {w > 0 ? w.toFixed(2) : '-'}
                </motion.div>
              ))}
            </motion.div>
          ))}
        </div>
      </div>

      <motion.p
        className="text-center text-slate-500 text-sm mt-4"
        animate={{
          opacity: isActive ? 1 : 0.5,
        }}
      >
        每个位置只能看到自己和之前的内容（因果掩码）
      </motion.p>
    </div>
  );
}
