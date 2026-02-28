import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

export default function PositionalEncodingToggle() {
  const [showPositional, setShowPositional] = useState(true);
  const [selectedDim, setSelectedDim] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const maxLen = 50;
  const dModel = 16;
  const [animProgress, setAnimProgress] = useState(1);

  // Calculate sin/cos positional encoding
  const getPositionalEncoding = (pos: number, dim: number): number => {
    const i = Math.floor(dim / 2);
    const angle = pos / Math.pow(10000, (2 * i) / dModel);
    return dim % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  };

  // Animated draw visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const cellWidth = width / maxLen;
    const cellHeight = height / dModel;

    let startTime: number;
    const duration = 800; // Animation duration in ms

    const draw = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      const progress = Math.min(elapsed / duration, 1);
      setAnimProgress(progress);

      ctx.clearRect(0, 0, width, height);

      // Draw heatmap with animation
      for (let pos = 0; pos < maxLen; pos++) {
        for (let dim = 0; dim < dModel; dim++) {
          // Animate column by column
          const columnProgress = Math.max(0, Math.min(1, (progress * maxLen - pos) / 2));
          const value = showPositional ? getPositionalEncoding(pos, dim) * columnProgress : 0;

          ctx.fillStyle = showPositional
            ? `rgba(${Math.abs(value) * 50 + 20}, ${Math.abs(value) * 150 + 50}, ${Math.abs(value) * 200 + 55}, ${columnProgress})`
            : 'rgb(30, 30, 40)';

          ctx.fillRect(pos * cellWidth, dim * cellHeight, cellWidth - 1, cellHeight - 1);
        }
      }

      // Highlight selected dimension
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.strokeRect(0, selectedDim * cellHeight, width, cellHeight);

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(draw);
      }
    };

    animationRef.current = requestAnimationFrame(draw);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [showPositional, selectedDim]);

  // Draw waveform for selected dimension with animation
  const waveformCanvasRef = useRef<HTMLCanvasElement>(null);
  const waveformAnimRef = useRef<number>();

  useEffect(() => {
    const canvas = waveformCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    let startTime: number;
    const duration = 600;

    const drawWaveform = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      const progress = Math.min(elapsed / duration, 1);

      ctx.clearRect(0, 0, width, height);

      // Draw grid
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();

      if (!showPositional) {
        ctx.fillStyle = '#64748b';
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('无位置编码 - 所有位置信息相同', width / 2, height / 2 + 5);
        return;
      }

      // Draw waveform with animation
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#06b6d4';
      ctx.shadowBlur = 10;
      ctx.beginPath();

      const drawLength = Math.floor(maxLen * progress);

      for (let pos = 0; pos < drawLength; pos++) {
        const value = getPositionalEncoding(pos, selectedDim);
        const x = (pos / maxLen) * width;
        const y = height / 2 - value * (height / 2 - 10);

        if (pos === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw animated dot at the end
      if (drawLength > 0 && progress < 1) {
        const lastPos = drawLength - 1;
        const lastValue = getPositionalEncoding(lastPos, selectedDim);
        const x = (lastPos / maxLen) * width;
        const y = height / 2 - lastValue * (height / 2 - 10);

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#06b6d4';
        ctx.fill();
      }

      // Label
      ctx.fillStyle = '#94a3b8';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'left';
      const freq = Math.pow(10000, Math.floor(selectedDim / 2) * 2 / dModel).toFixed(2);
      ctx.fillText(`维度 ${selectedDim}, 频率因子: ${freq}`, 10, 20);

      if (progress < 1) {
        waveformAnimRef.current = requestAnimationFrame(drawWaveform);
      }
    };

    waveformAnimRef.current = requestAnimationFrame(drawWaveform);

    return () => {
      if (waveformAnimRef.current) {
        cancelAnimationFrame(waveformAnimRef.current);
      }
    };
  }, [showPositional, selectedDim]);

  const handleToggle = () => {
    setIsAnimating(true);
    setShowPositional(!showPositional);
    setTimeout(() => setIsAnimating(false), 800);
  };

  return (
    <div className="visualization-container">
      <h3 className="text-lg font-bold text-cyan-400 mb-4">位置编码可视化</h3>

      {/* Toggle button with animation */}
      <div className="flex items-center gap-4 mb-6">
        <motion.button
          onClick={handleToggle}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className={`relative px-6 py-2 rounded-lg font-medium overflow-hidden ${
            showPositional
              ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/30'
              : 'bg-slate-700 text-slate-300'
          }`}
        >
          <motion.div
            className="absolute inset-0 bg-white/20"
            initial={{ x: '-100%' }}
            whileHover={{ x: '100%' }}
            transition={{ duration: 0.3 }}
          />
          <span className="relative z-10">
            {showPositional ? '✓ 位置编码开启' : '位置编码关闭'}
          </span>
        </motion.button>
        <span className="text-slate-400 text-sm">
          切换查看有无位置编码的差异
        </span>
      </div>

      {/* Heatmap */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-slate-400 text-sm">位置 →</span>
          <motion.span
            className="text-cyan-400 text-xs ml-auto"
            animate={{ opacity: isAnimating ? 1 : 0 }}
          >
            绘制中...
          </motion.span>
        </div>
        <motion.canvas
          ref={canvasRef}
          width={500}
          height={200}
          className="w-full cursor-pointer rounded-lg border border-slate-700"
          whileHover={{ borderColor: '#06b6d4' }}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const y = e.clientY - rect.top;
            const dim = Math.floor((y / rect.height) * dModel);
            setSelectedDim(Math.min(dim, dModel - 1));
          }}
        />
        <div className="flex justify-between text-xs text-slate-500 mt-1">
          <span>pos=0</span>
          <span>维度 ↓</span>
          <span>pos={maxLen}</span>
        </div>
      </div>

      {/* Waveform for selected dimension */}
      <div className="mt-6">
        <h4 className="text-sm font-medium text-slate-400 mb-2">
          维度 {selectedDim} 的波形
        </h4>
        <motion.canvas
          ref={waveformCanvasRef}
          width={500}
          height={100}
          className="w-full rounded-lg bg-slate-900 border border-slate-700"
          whileHover={{ borderColor: '#06b6d4' }}
        />
      </div>

      {/* Summary */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-6 p-4 bg-slate-800/50 rounded-lg border border-cyan-500/20"
      >
        <p className="text-slate-300 text-sm">
          <strong className="text-cyan-400">观察：</strong>
          {showPositional
            ? '每个位置都有独特的编码模式。低维度（上方）变化快，高维度（下方）变化慢。'
            : '没有位置编码，所有位置的表示完全相同，模型无法区分顺序。'}
        </p>
      </motion.div>
    </div>
  );
}
