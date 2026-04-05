import { useState, useMemo } from 'react';

export default function AttentionExplorer() {
  const [tokens, setTokens] = useState(['我', '爱', 'AI', '学习']);
  const [inputText, setInputText] = useState('我爱AI学习');
  const [selectedCell, setSelectedCell] = useState<{i: number; j: number} | null>(null);

  // Parse input
  const handleParse = () => {
    const newTokens = inputText.split('').filter(c => c.trim());
    if (newTokens.length > 0 && newTokens.length <= 8) {
      setTokens(newTokens);
      setSelectedCell(null);
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
      <h3 className="text-lg font-bold text-cyan-400 mb-4">Attention Explorer</h3>

      {/* Input */}
      <div className="flex gap-2 mb-6">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="输入文本（最多8个字符）"
          className="flex-1 px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-100"
        />
        <button
          onClick={handleParse}
          className="px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-500"
        >
          分析
        </button>
      </div>

      {/* Attention Matrix */}
      <div className="mb-6">
        <div className="flex justify-center">
          <div>
            {/* Column headers */}
            <div className="flex">
              <div className="w-16 h-8"></div>
              {tokens.map((t, i) => (
                <div key={i} className="w-16 h-8 flex items-center justify-center text-slate-400 text-sm">
                  {t}
                </div>
              ))}
            </div>

            {/* Rows */}
            {attentionWeights.map((row, i) => (
              <div key={i} className="flex">
                <div className="w-16 h-12 flex items-center justify-center text-slate-400 text-sm">
                  {tokens[i]}
                </div>
                {row.map((w, j) => (
                  <div
                    key={j}
                    className={`w-16 h-12 flex items-center justify-center cursor-pointer transition-all ${
                      selectedCell?.i === i && selectedCell?.j === j
                        ? 'ring-2 ring-amber-400'
                        : ''
                    }`}
                    style={{ backgroundColor: getColor(w) }}
                    onClick={() => setSelectedCell({ i, j })}
                  >
                    <span className="text-sm font-mono text-slate-100">
                      {w.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        <div className="text-center text-slate-500 text-xs mt-2">
          行 = Query (谁在看) | 列 = Key (被谁看到) | 颜色越亮 = 注意力越高
        </div>
      </div>

      {/* Selected cell explanation */}
      {selectedCell && (
        <div className="bg-slate-800 p-4 rounded-lg mb-4">
          <p className="text-slate-300">
            <span className="text-cyan-400 font-bold">{tokens[selectedCell.i]}</span>
            {' 对 '}
            <span className="text-green-400 font-bold">{tokens[selectedCell.j]}</span>
            {' 的注意力权重为 '}
            <span className="text-amber-400 font-bold">
              {(attentionWeights[selectedCell.i][selectedCell.j] * 100).toFixed(1)}%
            </span>
          </p>
          <p className="text-slate-500 text-sm mt-2">
            这意味着在计算 "{tokens[selectedCell.i]}" 的输出时，
            有 {(attentionWeights[selectedCell.i][selectedCell.j] * 100).toFixed(1)}% 的信息来自 "{tokens[selectedCell.j]}"
          </p>
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-4 text-sm text-slate-400">
        <span>注意力强度:</span>
        <div className="flex gap-1">
          {[0.1, 0.3, 0.5, 0.7, 0.9].map(v => (
            <div
              key={v}
              className="w-8 h-6"
              style={{ backgroundColor: getColor(v) }}
            />
          ))}
        </div>
        <span>低 → 高</span>
      </div>
    </div>
  );
}
