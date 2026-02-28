import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import PyodideRunner from './PyodideRunner';
import AnimatedCard from '../visualizations/AnimatedCard';
import GlowingButton from '../visualizations/GlowingButton';

type LabTab = 'playground' | 'attention' | 'training' | 'architecture';

interface LabPageProps {
  pyodideStatus?: 'loading' | 'ready' | 'error' | 'fallback' | 'idle';
}

export default function LabPage({ pyodideStatus = 'idle' }: LabPageProps) {
  const [activeTab, setActiveTab] = useState<LabTab>('playground');
  const [attentionData, setAttentionData] = useState<number[][][]>([]);
  const [internalStatus, setInternalStatus] = useState<'loading' | 'ready' | 'error' | 'fallback' | 'idle'>(pyodideStatus);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Handle status updates from PyodideRunner
  const handleStatusChange = (status: 'loading' | 'ready' | 'error' | 'fallback' | 'idle') => {
    setInternalStatus(status);
  };

  // Draw attention visualization
  useEffect(() => {
    if (!canvasRef.current || attentionData.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Draw attention heatmap
    const lastAttention = attentionData[attentionData.length - 1];
    if (!lastAttention) return;

    const seqLen = lastAttention.length;
    const cellSize = Math.min(width / seqLen, height / seqLen);
    const offsetX = (width - cellSize * seqLen) / 2;
    const offsetY = (height - cellSize * seqLen) / 2;

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        const weight = lastAttention[i][j];
        const intensity = Math.min(weight * 5, 1);

        ctx.fillStyle = `rgba(6, 182, 212, ${intensity})`;
        ctx.fillRect(
          offsetX + j * cellSize,
          offsetY + i * cellSize,
          cellSize - 1,
          cellSize - 1
        );
      }
    }
  }, [attentionData]);

  const tabs: { id: LabTab; label: string; icon: string }[] = [
    { id: 'playground', label: '游乐场', icon: '🎮' },
    { id: 'attention', label: 'Attention', icon: '🔍' },
    { id: 'training', label: '训练', icon: '📈' },
    { id: 'architecture', label: '架构', icon: '🏗️' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <div className="border-b border-slate-800/50 backdrop-blur-xl bg-slate-900/50 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                🧪 Transformer 交互实验室
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                从零开始理解、训练和运行 Mini Transformer
              </p>
            </div>

            <div className="flex items-center gap-2">
              <motion.div
                className={`px-3 py-1 rounded-full text-xs font-medium ${
                  internalStatus === 'ready'
                    ? 'bg-green-500/10 border border-green-500/30 text-green-400'
                    : internalStatus === 'loading'
                    ? 'bg-amber-500/10 border border-amber-500/30 text-amber-400'
                    : internalStatus === 'error'
                    ? 'bg-red-500/10 border border-red-500/30 text-red-400'
                    : internalStatus === 'fallback'
                    ? 'bg-amber-500/10 border border-amber-500/30 text-amber-400'
                    : 'bg-slate-500/10 border border-slate-500/30 text-slate-400'
                }`}
                animate={internalStatus === 'loading' ? {
                  boxShadow: ['0 0 0px rgba(245, 158, 11, 0)', '0 0 10px rgba(245, 158, 11, 0.3)', '0 0 0px rgba(245, 158, 11, 0)'],
                } : internalStatus === 'ready' ? {
                  boxShadow: ['0 0 0px rgba(16, 185, 129, 0)', '0 0 10px rgba(16, 185, 129, 0.3)', '0 0 0px rgba(16, 185, 129, 0)'],
                } : {}}
                transition={{ duration: 2, repeat: Infinity }}
              >
                {internalStatus === 'ready' && '✓ Pyodide 就绪'}
                {internalStatus === 'loading' && '⏳ Pyodide 加载中...'}
                {internalStatus === 'error' && '✗ Pyodide 加载失败'}
                {internalStatus === 'fallback' && '⚠ 简化模式'}
                {internalStatus === 'idle' && '○ 等待初始化'}
              </motion.div>
            </div>
          </div>

          {/* Tab navigation */}
          <div className="flex gap-1 mt-4">
            {tabs.map((tab) => (
              <motion.button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === tab.id
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                    : 'text-slate-400 hover:text-slate-300 hover:bg-slate-800/50'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <span className="mr-1.5">{tab.icon}</span>
                {tab.label}
              </motion.button>
            ))}
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-6xl mx-auto px-4 py-6">
        <AnimatePresence mode="wait">
          {activeTab === 'playground' && (
            <motion.div
              key="playground"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <AnimatedCard
                title="Mini Transformer"
                icon={<span>🤖</span>}
                delay={0.1}
              >
                <PyodideRunner
                  onAttentionUpdate={setAttentionData}
                  onStatusChange={handleStatusChange}
                />
              </AnimatedCard>

              {/* Real-time attention visualization */}
              {attentionData.length > 0 && (
                <AnimatedCard
                  title="实时 Attention 可视化"
                  icon={<span>🔍</span>}
                  delay={0.2}
                >
                  <div className="flex justify-center">
                    <canvas
                      ref={canvasRef}
                      width={300}
                      height={300}
                      className="rounded-lg bg-slate-950"
                    />
                  </div>
                  <p className="text-center text-slate-500 text-sm mt-3">
                    颜色越亮表示注意力权重越高
                  </p>
                </AnimatedCard>
              )}
            </motion.div>
          )}

          {activeTab === 'attention' && (
            <motion.div
              key="attention"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid grid-cols-1 lg:grid-cols-2 gap-6"
            >
              <AnimatedCard title="Self-Attention 原理" icon={<span>📐</span>}>
                <div className="space-y-4 text-slate-300 text-sm">
                  <p>
                    Self-Attention 是 Transformer 的核心机制，它允许每个位置直接关注序列中的所有其他位置。
                  </p>

                  <div className="p-3 bg-slate-800/50 rounded-lg font-mono text-xs">
                    <p className="text-cyan-400 mb-2"># Attention 公式</p>
                    <p>Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>)V</p>
                  </div>

                  <ul className="space-y-2">
                    <li className="flex items-start gap-2">
                      <span className="text-amber-400">Q</span>
                      <span>Query - "我在找什么信息？"</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400">K</span>
                      <span>Key - "我有什么信息？"</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-purple-400">V</span>
                      <span>Value - "我要传递的内容"</span>
                    </li>
                  </ul>
                </div>
              </AnimatedCard>

              <AnimatedCard title="交互式 Attention 矩阵" icon={<span>🔥</span>} delay={0.1}>
                <div className="aspect-square bg-slate-950 rounded-lg flex items-center justify-center">
                  <p className="text-slate-500 text-sm">在"游乐场"运行模型后这里会显示 Attention 矩阵</p>
                </div>
              </AnimatedCard>
            </motion.div>
          )}

          {activeTab === 'training' && (
            <motion.div
              key="training"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <AnimatedCard title="训练监控" icon={<span>📊</span>}>
                <div className="h-64 bg-slate-950 rounded-lg flex items-center justify-center">
                  <p className="text-slate-500 text-sm">训练开始后会显示 Loss 曲线</p>
                </div>
              </AnimatedCard>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <AnimatedCard delay={0.1}>
                  <div className="text-center">
                    <p className="text-3xl font-bold text-cyan-400">0</p>
                    <p className="text-slate-500 text-sm mt-1">训练轮数</p>
                  </div>
                </AnimatedCard>

                <AnimatedCard delay={0.2}>
                  <div className="text-center">
                    <p className="text-3xl font-bold text-green-400">0.000</p>
                    <p className="text-slate-500 text-sm mt-1">当前 Loss</p>
                  </div>
                </AnimatedCard>

                <AnimatedCard delay={0.3}>
                  <div className="text-center">
                    <p className="text-3xl font-bold text-purple-400">0</p>
                    <p className="text-slate-500 text-sm mt-1">参数数量</p>
                  </div>
                </AnimatedCard>
              </div>
            </motion.div>
          )}

          {activeTab === 'architecture' && (
            <motion.div
              key="architecture"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <AnimatedCard title="模型架构" icon={<span>🏗️</span>}>
                <div className="space-y-4">
                  {/* Architecture diagram */}
                  <div className="flex flex-col items-center gap-4">
                    {[
                      { name: 'Input Embedding', color: 'from-cyan-500/20 to-blue-500/20' },
                      { name: 'Positional Encoding', color: 'from-green-500/20 to-emerald-500/20' },
                      { name: 'Multi-Head Attention', color: 'from-amber-500/20 to-orange-500/20' },
                      { name: 'Add & Norm', color: 'from-purple-500/20 to-pink-500/20' },
                      { name: 'Feed Forward', color: 'from-red-500/20 to-rose-500/20' },
                      { name: 'Output', color: 'from-cyan-500/20 to-blue-500/20' },
                    ].map((layer, i) => (
                      <motion.div
                        key={layer.name}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className={`w-64 py-3 px-4 rounded-lg bg-gradient-to-r ${layer.color}
                          border border-slate-700/50 text-center text-sm font-medium text-slate-200`}
                      >
                        {layer.name}
                      </motion.div>
                    ))}
                  </div>

                  <div className="p-4 bg-slate-800/30 rounded-lg">
                    <h4 className="text-sm font-medium text-slate-300 mb-2">模型配置</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-slate-500">d_model:</span>
                        <span className="text-cyan-400 font-mono">32</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">n_heads:</span>
                        <span className="text-cyan-400 font-mono">2</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">vocab_size:</span>
                        <span className="text-cyan-400 font-mono">动态</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">max_seq_len:</span>
                        <span className="text-cyan-400 font-mono">512</span>
                      </div>
                    </div>
                  </div>
                </div>
              </AnimatedCard>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
