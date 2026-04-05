import { useState, useMemo } from 'react';

export default function MultiHeadComparator() {
  const [numHeads, setNumHeads] = useState(4);
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
            // Different heads focus on different patterns
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

  const renderAttentionMatrix = (weights: number[][], title: string, compact = false) => (
    <div className="text-center">
      <div className="text-xs text-slate-400 mb-1">{title}</div>
      <div className="inline-block">
        {weights.map((row, i) => (
          <div key={i} className="flex">
            {row.map((w, j) => (
              <div
                key={j}
                className={`${compact ? 'w-6 h-6' : 'w-8 h-8'} flex items-center justify-center text-xs`}
                style={{ backgroundColor: getColor(w) }}
              >
                {compact ? '' : w.toFixed(1)}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="visualization-container">
      <h3 className="text-lg font-bold text-cyan-400 mb-4">Multi-Head Attention 对比器</h3>

      {/* Head selector */}
      <div className="mb-6">
        <label className="block text-slate-400 mb-2">注意力头数量: {numHeads}</label>
        <input
          type="range"
          min={1}
          max={8}
          value={numHeads}
          onChange={(e) => setNumHeads(parseInt(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-slate-500">
          <span>1 (单头)</span>
          <span>8 (多头)</span>
        </div>
      </div>

      {/* Single head comparison */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-slate-800/50 p-4 rounded-lg">
          <h4 className="text-center font-bold text-amber-400 mb-4">单头 Attention</h4>
          {renderAttentionMatrix(headPatterns[1][0], '1 个头')}
          <p className="text-xs text-slate-500 mt-3 text-center">
            只能学习一种注意力模式
          </p>
        </div>

        <div className="bg-slate-800/50 p-4 rounded-lg">
          <h4 className="text-center font-bold text-green-400 mb-4">{numHeads} 头 Attention (合并后)</h4>
          {renderAttentionMatrix(getMergedAttention(numHeads), `${numHeads} 个头平均`)}
          <p className="text-xs text-slate-500 mt-3 text-center">
            多个头的模式融合在一起
          </p>
        </div>
      </div>

      {/* Individual heads */}
      {numHeads > 1 && (
        <div className="mb-6">
          <h4 className="text-sm font-bold text-slate-400 mb-3">各个头的独立模式:</h4>
          <div className="grid grid-cols-4 gap-2">
            {headPatterns[numHeads].slice(0, numHeads).map((weights, idx) => (
              <div key={idx} className="bg-slate-800/30 p-2 rounded">
                {renderAttentionMatrix(weights, `Head ${idx + 1}`, true)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Explanation */}
      <div className="bg-slate-800/50 p-4 rounded-lg">
        <h4 className="font-bold text-cyan-400 mb-2">观察要点</h4>
        <ul className="text-sm text-slate-300 space-y-1">
          <li>• <strong>单头</strong>：只能捕获一种注意力模式</li>
          <li>• <strong>多头</strong>：每个头可以学习不同的模式（局部、全局、语法关系等）</li>
          <li>• <strong>合并后</strong>：多个模式融合，表达能力更强</li>
        </ul>
      </div>
    </div>
  );
}
