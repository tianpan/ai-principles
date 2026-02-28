import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function QKVStepByStep() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isAutoPlay, setIsAutoPlay] = useState(false);

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

  // Auto-play effect
  useEffect(() => {
    if (!isAutoPlay) return;

    const timer = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= 4) {
          setIsAutoPlay(false);
          return 4;
        }
        return prev + 1;
      });
    }, 2500);

    return () => clearInterval(timer);
  }, [isAutoPlay]);

  const steps = [
    {
      title: 'Step 1: 输入 Embedding',
      description: '每个 token 被转换为向量',
      visualization: (
        <div className="flex gap-6 justify-center">
          {tokens.map((token, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.15 }}
              className="text-center"
            >
              <motion.div
                className="text-lg font-bold text-cyan-400 mb-2"
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 2, repeat: Infinity, delay: i * 0.3 }}
              >
                {token}
              </motion.div>
              <div className="bg-slate-800 p-3 rounded-lg border border-cyan-500/20">
                {embeddings[i].map((v, j) => (
                  <motion.div
                    key={j}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.15 + j * 0.1 }}
                    className="text-sm font-mono text-slate-300 flex items-center gap-2"
                  >
                    <span className="w-4 text-slate-500 text-xs">d{j}</span>
                    <motion.span
                      className="text-cyan-300"
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 1.5, repeat: Infinity, delay: j * 0.2 }}
                    >
                      {v.toFixed(1)}
                    </motion.span>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      ),
    },
    {
      title: 'Step 2: 计算 Q, K, V',
      description: '通过线性变换得到 Query, Key, Value',
      visualization: (
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: 'Query (Q)', data: Q, color: 'amber', desc: '我在找什么' },
            { label: 'Key (K)', data: K, color: 'green', desc: '我有什么信息' },
            { label: 'Value (V)', data: V, color: 'purple', desc: '实际传递的内容' },
          ].map((item, idx) => (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.2 }}
            >
              <div className={`text-center font-bold text-${item.color}-400 mb-2`}>
                {item.label}
              </div>
              <div className={`text-center text-xs text-slate-500 mb-2`}>{item.desc}</div>
              {item.data.map((vec, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.2 + i * 0.1 }}
                  className={`bg-slate-800 p-2 rounded mb-1 text-xs font-mono border border-${item.color}-500/20`}
                >
                  [{vec.map(v => v.toFixed(2)).join(', ')}]
                </motion.div>
              ))}
            </motion.div>
          ))}
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
                  <motion.div
                    key={i}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.1 }}
                    className="w-16 text-center text-xs text-slate-400"
                  >
                    {t}
                  </motion.div>
                ))}
              </div>
              {scores.map((row, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 + i * 0.1 }}
                  className="flex gap-2 items-center"
                >
                  <div className="w-12 text-xs text-slate-400">{tokens[i]}</div>
                  {row.map((score, j) => (
                    <motion.div
                      key={j}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.3 + i * 0.1 + j * 0.05, type: "spring" }}
                      className="w-16 text-center text-sm font-mono rounded"
                      style={{
                        backgroundColor: `rgba(6, 182, 212, ${Math.abs(score) / 3})`,
                      }}
                    >
                      {(score / Math.sqrt(dk)).toFixed(2)}
                    </motion.div>
                  ))}
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      ),
    },
    {
      title: 'Step 4: Softmax 归一化',
      description: '将分数转换为概率分布（每行和为1）',
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
                <motion.div
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.15 }}
                  className="flex gap-2 items-center"
                >
                  <div className="w-12 text-xs text-slate-400">{tokens[i]}</div>
                  {row.map((w, j) => (
                    <motion.div
                      key={j}
                      initial={{ scale: 0, backgroundColor: 'rgba(16, 185, 129, 0)' }}
                      animate={{
                        scale: 1,
                        backgroundColor: `rgba(16, 185, 129, ${w})`
                      }}
                      transition={{ delay: i * 0.15 + j * 0.1, duration: 0.3 }}
                      className="w-16 text-center text-sm font-mono rounded"
                    >
                      {w.toFixed(2)}
                    </motion.div>
                  ))}
                </motion.div>
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
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.2, type: "spring" }}
                className="text-center"
              >
                <motion.div
                  className="text-lg font-bold text-cyan-400 mb-2"
                  animate={{
                    textShadow: [
                      '0 0 0px rgba(6, 182, 212, 0)',
                      '0 0 10px rgba(6, 182, 212, 0.5)',
                      '0 0 0px rgba(6, 182, 212, 0)'
                    ]
                  }}
                  transition={{ duration: 2, repeat: Infinity, delay: i * 0.3 }}
                >
                  {token}
                </motion.div>
                <div className="bg-slate-800 p-3 rounded-lg border border-green-500/30 shadow-lg shadow-green-500/10">
                  {outputs[i].map((v, j) => (
                    <motion.div
                      key={j}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.2 + j * 0.1 }}
                      className="text-sm font-mono text-green-400"
                    >
                      {v.toFixed(2)}
                    </motion.div>
                  ))}
                </div>
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.2 + 0.3 }}
                  className="text-xs text-slate-500 mt-1"
                >
                  输出向量
                </motion.div>
              </motion.div>
            ))}
          </div>
        </div>
      ),
    },
  ];

  return (
    <div className="visualization-container">
      <h3 className="text-lg font-bold text-cyan-400 mb-4">Self-Attention 分步演示</h3>

      {/* Progress bar with animation */}
      <div className="flex items-center gap-2 mb-6">
        {steps.map((_, i) => (
          <motion.div
            key={i}
            className={`h-2 flex-1 rounded cursor-pointer ${
              i <= currentStep ? 'bg-cyan-500' : 'bg-slate-700'
            }`}
            whileHover={{ scaleY: 1.5 }}
            onClick={() => setCurrentStep(i)}
          />
        ))}
      </div>

      {/* Current step with transition */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="mb-6"
        >
          <h4 className="text-xl font-bold text-slate-100 mb-2">
            {steps[currentStep].title}
          </h4>
          <p className="text-slate-400 mb-4">{steps[currentStep].description}</p>
          {steps[currentStep].visualization}
        </motion.div>
      </AnimatePresence>

      {/* Navigation */}
      <div className="flex justify-between items-center">
        <motion.button
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="px-4 py-2 rounded-lg bg-slate-700 text-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ← 上一步
        </motion.button>

        <div className="flex items-center gap-3">
          <span className="text-slate-500">
            {currentStep + 1} / {steps.length}
          </span>
          <motion.button
            onClick={() => setIsAutoPlay(!isAutoPlay)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`px-4 py-2 rounded-lg font-medium ${
              isAutoPlay
                ? 'bg-amber-500 text-white'
                : 'bg-slate-700 text-slate-300'
            }`}
          >
            {isAutoPlay ? '⏸ 暂停' : '▶ 自动播放'}
          </motion.button>
        </div>

        <motion.button
          onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
          disabled={currentStep === steps.length - 1}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"
        >
          下一步 →
        </motion.button>
      </div>
    </div>
  );
}
