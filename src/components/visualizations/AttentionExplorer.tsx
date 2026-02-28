import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function AttentionExplorer() {
  const [tokens, setTokens] = useState(['我', '爱', 'AI', '学习']);
  const [inputText, setInputText] = useState('我爱AI学习');
  const [selectedCell, setSelectedCell] = useState<{i: number; j: number} | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  // Parse input
  const handleParse = () => {
    const newTokens = inputText.split('').filter(c => c.trim());
    if (newTokens.length > 0 && newTokens.length <= 8) {
      setIsAnimating(true);
      setTokens(newTokens);
      setSelectedCell(null);
      setTimeout(() => setIsAnimating(false), 500);
    }
  };

  // Simulate attention weights (for demo)
  const attentionWeights = useMemo(() => {
    const n = tokens.length;
    const weights: number[][] = [];

    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      let sum = 0;

      for (let j = 0; j < n; j++) {
        // Simulate: closer tokens get higher attention
        const dist = Math.abs(i - j);
        const base = Math.exp(-dist * 0.5);
        // Add some randomness
        const noise = Math.random() * 0.2;
        row.push(base + noise);
        sum += base + noise;
      }

      // Normalize
      weights.push(row.map(w => w / sum));
    }

    return weights;
  }, [tokens]);

  // Get color for weight
  const getColor = (weight: number) => {
    const intensity = Math.min(weight * 3, 1);
    return `rgba(6, 182, 212, ${intensity})`;
  };

  return (
    <div className="visualization-container">
      <h3 className="text-lg font-bold text-cyan-400 mb-4">
        Attention Explorer
        <motion.span
          className="ml-2 text-sm text-slate-500"
          animate={{ opacity: isAnimating ? 1 : 0 }}
        >
          计算中...
        </motion.span>
      </h3>

      {/* Input */}
      <div className="flex gap-2 mb-6">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="输入文本（最多8个字符）"
          className="flex-1 px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-100 focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all"
        />
        <motion.button
          onClick={handleParse}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="px-6 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-lg font-medium shadow-lg shadow-cyan-500/20"
        >
          分析
        </motion.button>
      </div>

      {/* Attention Matrix with Animation */}
      <div className="mb-6">
        <div className="flex justify-center">
          <div>
            {/* Column headers */}
            <div className="flex">
              <div className="w-16 h-8"></div>
              {tokens.map((t, i) => (
                <motion.div
                  key={`${t}-${i}`}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1, duration: 0.3 }}
                  className="w-16 h-8 flex items-center justify-center text-slate-400 text-sm font-medium"
                >
                  {t}
                </motion.div>
              ))}
            </div>

            {/* Rows with animated cells */}
            {attentionWeights.map((row, i) => (
              <div key={i} className="flex">
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1, duration: 0.3 }}
                  className="w-16 h-12 flex items-center justify-center text-slate-400 text-sm font-medium"
                >
                  {tokens[i]}
                </motion.div>
                {row.map((w, j) => (
                  <motion.div
                    key={`${i}-${j}`}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{
                      delay: i * 0.1 + j * 0.05,
                      type: "spring",
                      stiffness: 200,
                      damping: 15
                    }}
                    whileHover={{ scale: 1.1, zIndex: 10 }}
                    className={`w-16 h-12 flex items-center justify-center cursor-pointer transition-all rounded m-0.5 ${
                      selectedCell?.i === i && selectedCell?.j === j
                        ? 'ring-2 ring-amber-400 shadow-lg shadow-amber-500/30'
                        : ''
                    }`}
                    style={{ backgroundColor: getColor(w) }}
                    onClick={() => setSelectedCell({ i, j })}
                  >
                    <span className="text-sm font-mono text-slate-100">
                      {w.toFixed(2)}
                    </span>
                  </motion.div>
                ))}
              </div>
            ))}
          </div>
        </div>

        <div className="text-center text-slate-500 text-xs mt-2">
          行 = Query (谁在看) | 列 = Key (被谁看到) | 颜色越亮 = 注意力越高
        </div>
      </div>

      {/* Animated connection lines when cell selected */}
      <AnimatePresence>
        {selectedCell && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-slate-800 p-4 rounded-lg mb-4 border border-cyan-500/20"
          >
            <div className="flex items-center justify-center gap-4 mb-3">
              {/* Query token */}
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-cyan-500/30"
              >
                {tokens[selectedCell.i]}
              </motion.div>

              {/* Animated arrow */}
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: 40 }}
                transition={{ delay: 0.2 }}
                className="flex items-center"
              >
                <div className="h-0.5 bg-gradient-to-r from-cyan-500 to-green-500 flex-1"></div>
                <div className="w-0 h-0 border-t-4 border-b-4 border-l-6 border-transparent border-l-green-500"></div>
              </motion.div>

              {/* Key token */}
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.3 }}
                className="w-12 h-12 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-green-500/30"
              >
                {tokens[selectedCell.j]}
              </motion.div>

              {/* Weight badge */}
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.4, type: "spring" }}
                className="px-3 py-1 rounded-full bg-amber-500/20 border border-amber-500/50 text-amber-400 font-bold"
              >
                {(attentionWeights[selectedCell.i][selectedCell.j] * 100).toFixed(1)}%
              </motion.div>
            </div>

            <p className="text-slate-500 text-sm text-center">
              这意味着在计算 "{tokens[selectedCell.i]}" 的输出时，
              有 {(attentionWeights[selectedCell.i][selectedCell.j] * 100).toFixed(1)}% 的信息来自 "{tokens[selectedCell.j]}"
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Legend */}
      <div className="flex items-center gap-4 text-sm text-slate-400">
        <span>注意力强度:</span>
        <div className="flex gap-1">
          {[0.1, 0.3, 0.5, 0.7, 0.9].map((v, i) => (
            <motion.div
              key={v}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: i * 0.1 }}
              className="w-8 h-6 rounded"
              style={{ backgroundColor: getColor(v) }}
            />
          ))}
        </div>
        <span>低 → 高</span>
      </div>
    </div>
  );
}
