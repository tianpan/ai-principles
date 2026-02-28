import { useState } from 'react';

export default function QKVStepByStep() {
  const [currentStep, setCurrentStep] = useState(0);

  // Example input
  const tokens = ['我', '爱', 'AI'];
  const embeddings = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
  ];

  // Simplified Q, K, V (identity for demo)
  const Q = embeddings.map(e => e.map(v => v * 1.1));
  const K = embeddings.map(e => e.map(v => v * 0.9));
  const V = embeddings.map(e => e.map(v => v * 1.0));

  // Calculate attention scores
  const calculateScores = (q: number[], k: number[]) =>
    q.reduce((sum, qi, i) => sum + qi * k[i], 0);

  const scores = Q.map(q => K.map(k => calculateScores(q, k)));
  const dk = 3;

  // Softmax
  const softmax = (arr: number[]) => {
    const exp = arr.map(a => Math.exp(a));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(e => e / sum);
  };

  const attentionWeights = scores.map(row => softmax(row.map(s => s / Math.sqrt(dk))));

  // Weighted sum of V
  const outputs = attentionWeights.map(weights =>
    V[0].map((_, dim) =>
      weights.reduce((sum, w, i) => sum + w * V[i][dim], 0)
    )
  );

  const steps = [
    {
      title: 'Step 1: 输入 Embedding',
      description: '每个 token 被转换为向量',
      visualization: (
        <div className="flex gap-6 justify-center">
          {tokens.map((token, i) => (
            <div key={i} className="text-center">
              <div className="text-lg font-bold text-cyan-400 mb-2">{token}</div>
              <div className="bg-slate-800 p-3 rounded-lg">
                {embeddings[i].map((v, j) => (
                  <div key={j} className="text-sm font-mono text-slate-300">
                    {v.toFixed(1)}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      ),
    },
    {
      title: 'Step 2: 计算 Q, K, V',
      description: '通过线性变换得到 Query, Key, Value',
      visualization: (
        <div className="grid grid-cols-3 gap-4">
          <div>
            <div className="text-center font-bold text-amber-400 mb-2">Query (Q)</div>
            {Q.map((q, i) => (
              <div key={i} className="bg-slate-800 p-2 rounded mb-1 text-xs font-mono">
                [{q.map(v => v.toFixed(2)).join(', ')}]
              </div>
            ))}
          </div>
          <div>
            <div className="text-center font-bold text-green-400 mb-2">Key (K)</div>
            {K.map((k, i) => (
              <div key={i} className="bg-slate-800 p-2 rounded mb-1 text-xs font-mono">
                [{k.map(v => v.toFixed(2)).join(', ')}]
              </div>
            ))}
          </div>
          <div>
            <div className="text-center font-bold text-purple-400 mb-2">Value (V)</div>
            {V.map((v, i) => (
              <div key={i} className="bg-slate-800 p-2 rounded mb-1 text-xs font-mono">
                [{v.map(val => val.toFixed(2)).join(', ')}]
              </div>
            ))}
          </div>
        </div>
      ),
    },
    {
      title: 'Step 3: 计算注意力分数',
      description: 'Q @ K^T / √d_k',
      visualization: (
        <div>
          <div className="text-center text-slate-400 mb-2">
            分数矩阵 (行=Query, 列=Key)
          </div>
          <div className="flex justify-center">
            <div className="bg-slate-800 p-4 rounded-lg">
              <div className="flex gap-2 mb-2">
                <div className="w-12"></div>
                {tokens.map((t, i) => (
                  <div key={i} className="w-16 text-center text-xs text-slate-400">{t}</div>
                ))}
              </div>
              {scores.map((row, i) => (
                <div key={i} className="flex gap-2 items-center">
                  <div className="w-12 text-xs text-slate-400">{tokens[i]}</div>
                  {row.map((score, j) => (
                    <div
                      key={j}
                      className="w-16 text-center text-sm font-mono rounded"
                      style={{
                        backgroundColor: `rgba(6, 182, 212, ${Math.abs(score) / 3})`,
                      }}
                    >
                      {(score / Math.sqrt(dk)).toFixed(2)}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      ),
    },
    {
      title: 'Step 4: Softmax 归一化',
      description: '将分数转换为概率分布',
      visualization: (
        <div>
          <div className="text-center text-slate-400 mb-2">
            注意力权重 (每行和为 1)
          </div>
          <div className="flex justify-center">
            <div className="bg-slate-800 p-4 rounded-lg">
              <div className="flex gap-2 mb-2">
                <div className="w-12"></div>
                {tokens.map((t, i) => (
                  <div key={i} className="w-16 text-center text-xs text-slate-400">{t}</div>
                ))}
              </div>
              {attentionWeights.map((row, i) => (
                <div key={i} className="flex gap-2 items-center">
                  <div className="w-12 text-xs text-slate-400">{tokens[i]}</div>
                  {row.map((w, j) => (
                    <div
                      key={j}
                      className="w-16 text-center text-sm font-mono rounded"
                      style={{
                        backgroundColor: `rgba(16, 185, 129, ${w})`,
                      }}
                    >
                      {w.toFixed(2)}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      ),
    },
    {
      title: 'Step 5: 加权求和',
      description: 'attention_weights @ V = 输出',
      visualization: (
        <div>
          <div className="text-center text-slate-400 mb-4">
            每个位置的输出向量
          </div>
          <div className="flex gap-6 justify-center">
            {tokens.map((token, i) => (
              <div key={i} className="text-center">
                <div className="text-lg font-bold text-cyan-400 mb-2">{token}</div>
                <div className="bg-slate-800 p-3 rounded-lg">
                  {outputs[i].map((v, j) => (
                    <div key={j} className="text-sm font-mono text-green-400">
                      {v.toFixed(2)}
                    </div>
                  ))}
                </div>
                <div className="text-xs text-slate-500 mt-1">输出向量</div>
              </div>
            ))}
          </div>
        </div>
      ),
    },
  ];

  return (
    <div className="visualization-container">
      <h3 className="text-lg font-bold text-cyan-400 mb-4">Self-Attention 分步演示</h3>

      {/* Progress bar */}
      <div className="flex items-center gap-2 mb-6">
        {steps.map((_, i) => (
          <div
            key={i}
            className={`h-2 flex-1 rounded ${
              i <= currentStep ? 'bg-cyan-500' : 'bg-slate-700'
            }`}
          />
        ))}
      </div>

      {/* Current step */}
      <div className="mb-6">
        <h4 className="text-xl font-bold text-slate-100 mb-2">
          {steps[currentStep].title}
        </h4>
        <p className="text-slate-400 mb-4">{steps[currentStep].description}</p>
        {steps[currentStep].visualization}
      </div>

      {/* Navigation */}
      <div className="flex justify-between items-center">
        <button
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          className="px-4 py-2 rounded-lg bg-slate-700 text-slate-300 disabled:opacity-50"
        >
          ← 上一步
        </button>
        <span className="text-slate-500">
          {currentStep + 1} / {steps.length}
        </span>
        <button
          onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
          disabled={currentStep === steps.length - 1}
          className="px-4 py-2 rounded-lg bg-cyan-600 text-white disabled:opacity-50"
        >
          下一步 →
        </button>
      </div>
    </div>
  );
}
