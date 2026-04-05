import { useState, useEffect, useRef } from 'react';

export default function PositionalEncodingToggle() {
  const [showPositional, setShowPositional] = useState(true);
  const [selectedDim, setSelectedDim] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maxLen = 50;
  const dModel = 16;

  // Calculate sin/cos positional encoding
  const getPositionalEncoding = (pos: number, dim: number): number => {
    const i = Math.floor(dim / 2);
    const angle = pos / Math.pow(10000, (2 * i) / dModel);
    return dim % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  };

  // Draw visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const cellWidth = width / maxLen;
    const cellHeight = height / dModel;

    ctx.clearRect(0, 0, width, height);

    // Draw heatmap
    for (let pos = 0; pos < maxLen; pos++) {
      for (let dim = 0; dim < dModel; dim++) {
        const value = showPositional ? getPositionalEncoding(pos, dim) : 0;
        // Map [-1, 1] to color
        const r = value > 0 ? Math.floor(value * 255 * 0.8) : 0;
        const g = value > 0 ? Math.floor(value * 255 * 0.8) : 0;
        const b = value < 0 ? Math.floor(-value * 255 * 0.8) : Math.floor(50 + value * 100);

        ctx.fillStyle = showPositional
          ? `rgb(${Math.abs(value) * 50 + 20}, ${Math.abs(value) * 150 + 50}, ${Math.abs(value) * 200 + 55})`
          : 'rgb(30, 30, 40)';

        ctx.fillRect(pos * cellWidth, dim * cellHeight, cellWidth - 1, cellHeight - 1);
      }
    }

    // Highlight selected dimension
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, selectedDim * cellHeight, width, cellHeight);

  }, [showPositional, selectedDim]);

  // Draw waveform for selected dimension
  const waveformCanvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = waveformCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

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

    // Draw waveform
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let pos = 0; pos < maxLen; pos++) {
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

    // Label
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'left';
    const freq = Math.pow(10000, Math.floor(selectedDim / 2) * 2 / dModel).toFixed(2);
    ctx.fillText(`维度 ${selectedDim}, 频率因子: ${freq}`, 10, 20);

  }, [showPositional, selectedDim]);

  return (
    <div className="visualization-container">
      <h3 className="text-lg font-bold text-cyan-400 mb-4">位置编码可视化</h3>

      {/* Toggle */}
      <div className="flex items-center gap-4 mb-6">
        <button
          onClick={() => setShowPositional(!showPositional)}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            showPositional
              ? 'bg-cyan-500 text-white'
              : 'bg-slate-700 text-slate-300'
          }`}
        >
          {showPositional ? '✓ 位置编码开启' : '位置编码关闭'}
        </button>
        <span className="text-slate-400 text-sm">
          切换查看有无位置编码的差异
        </span>
      </div>

      {/* Heatmap */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-slate-400 text-sm">位置 →</span>
          <span className="text-slate-500 text-xs ml-auto">点击行选择维度</span>
        </div>
        <canvas
          ref={canvasRef}
          width={500}
          height={200}
          className="w-full cursor-pointer rounded-lg"
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
        <canvas
          ref={waveformCanvasRef}
          width={500}
          height={100}
          className="w-full rounded-lg bg-slate-900"
        />
      </div>

      {/* Summary */}
      <div className="mt-6 p-4 bg-slate-800/50 rounded-lg">
        <p className="text-slate-300 text-sm">
          <strong className="text-cyan-400">观察：</strong>
          {showPositional
            ? '每个位置都有独特的编码模式。低维度（上方）变化快，高维度（下方）变化慢。'
            : '没有位置编码，所有位置的表示完全相同，模型无法区分顺序。'}
        </p>
      </div>
    </div>
  );
}
