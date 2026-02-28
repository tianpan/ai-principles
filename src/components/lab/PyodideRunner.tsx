import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import LoadingSpinner from '../visualizations/LoadingSpinner';
import GlowingButton from '../visualizations/GlowingButton';
import { useTransformer } from '../../hooks/useTransformer';
import {
  FallbackTransformer,
  shouldUseFallback,
  getFallbackStatusMessage
} from '../../utils/fallbackTransformer';

interface PyodideRunnerProps {
  onAttentionUpdate?: (attention: number[][][]) => void;
  onTrainingProgress?: (epoch: number, loss: number) => void;
  onStatusChange?: (status: 'loading' | 'ready' | 'error' | 'fallback' | 'idle') => void;
}

export default function PyodideRunner({ onAttentionUpdate, onTrainingProgress, onStatusChange }: PyodideRunnerProps) {
  const {
    status,
    progress,
    stage,
    error,
    trainingProgress,
    generationResult,
    isLoading,
    isReady,
    isTraining,
    isGenerating,
    hasError,
    trainModel,
    generateText,
    reset,
  } = useTransformer();

  const [inputText, setInputText] = useState('我爱AI');
  const [outputText, setOutputText] = useState('');
  const [trainingData, setTrainingData] = useState('我爱学习AI。AI改变世界。学习使人进步。');
  const [epochs, setEpochs] = useState(10);
  const [learningRate, setLearningRate] = useState(0.01);

  // Fallback mode state
  const [isFallbackMode, setIsFallbackMode] = useState(false);
  const fallbackRef = useRef<FallbackTransformer | null>(null);

  // Check if we should use fallback mode
  useEffect(() => {
    if (shouldUseFallback()) {
      setIsFallbackMode(true);
      fallbackRef.current = new FallbackTransformer();
      onStatusChange?.('fallback');
    }
  }, [onStatusChange]);

  // Update status when it changes
  useEffect(() => {
    if (isFallbackMode) {
      onStatusChange?.('fallback');
    } else if (hasError) {
      onStatusChange?.('error');
    } else if (isLoading) {
      onStatusChange?.('loading');
    } else if (isReady) {
      onStatusChange?.('ready');
    } else {
      onStatusChange?.('idle');
    }
  }, [isFallbackMode, hasError, isLoading, isReady, onStatusChange]);

  // Update attention visualization
  useEffect(() => {
    if (generationResult?.attention) {
      onAttentionUpdate?.(generationResult.attention);
    }
  }, [generationResult, onAttentionUpdate]);

  // Update training progress
  useEffect(() => {
    if (trainingProgress) {
      onTrainingProgress?.(trainingProgress.epoch, trainingProgress.loss);
    }
  }, [trainingProgress, onTrainingProgress]);

  // Update output when generation completes
  useEffect(() => {
    if (generationResult) {
      setOutputText(generationResult.text);
    }
  }, [generationResult]);

  // Fallback handlers
  const [fallbackTrainingProgress, setFallbackTrainingProgress] = useState<{ epoch: number; loss: number } | null>(null);
  const [fallbackIsTraining, setFallbackIsTraining] = useState(false);
  const [fallbackIsGenerating, setFallbackIsGenerating] = useState(false);

  const handleTrain = async () => {
    if (isFallbackMode && fallbackRef.current) {
      setFallbackIsTraining(true);
      setFallbackTrainingProgress(null);

      // Simulate progress updates
      for (let i = 1; i <= epochs; i++) {
        await new Promise(resolve => setTimeout(resolve, 100));
        const progress = fallbackRef.current!.getTrainingProgress(i, epochs);
        setFallbackTrainingProgress({ epoch: i, loss: progress.loss });
        onTrainingProgress?.(i, progress.loss);
      }

      await fallbackRef.current.train(trainingData, epochs, learningRate);
      setFallbackIsTraining(false);
      setFallbackTrainingProgress(null);
    } else {
      trainModel(trainingData, epochs, learningRate);
    }
  };

  const handleGenerate = async () => {
    if (isFallbackMode && fallbackRef.current) {
      setFallbackIsGenerating(true);
      const result = await fallbackRef.current.generate(inputText, 20, 0.8);
      setOutputText(result.text);
      onAttentionUpdate?.(result.attention);
      setFallbackIsGenerating(false);
    } else {
      generateText(inputText, 20, 0.8);
    }
  };

  const getStageText = (s: string) => {
    switch (s) {
      case 'loading':
        return '加载 Pyodide 运行时...';
      case 'initializing':
        return '初始化 Python 环境...';
      case 'ready':
        return '准备就绪';
      default:
        return s;
    }
  };

  // Determine effective states
  const effectiveIsLoading = !isFallbackMode && isLoading;
  const effectiveIsReady = isFallbackMode || isReady;
  const effectiveIsTraining = isFallbackMode ? fallbackIsTraining : isTraining;
  const effectiveIsGenerating = isFallbackMode ? fallbackIsGenerating : isGenerating;
  const effectiveHasError = !isFallbackMode && hasError;
  const effectiveTrainingProgress = isFallbackMode ? fallbackTrainingProgress : trainingProgress;

  return (
    <div className="bg-slate-900/80 backdrop-blur-xl rounded-xl border border-slate-700/50 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-700/50 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <span className="text-cyan-400">⚡</span>
          Mini Transformer 运行器
          {isFallbackMode && (
            <span className="ml-2 text-xs px-2 py-0.5 bg-amber-500/20 text-amber-400 rounded">
              简化模式
            </span>
          )}
        </h3>

        {/* Status indicator */}
        <div className="flex items-center gap-2">
          <motion.div
            className={`w-2 h-2 rounded-full ${
              effectiveIsLoading ? 'bg-amber-400' :
              effectiveIsTraining ? 'bg-green-400' :
              effectiveIsGenerating ? 'bg-purple-400' :
              effectiveIsReady ? 'bg-cyan-400' :
              effectiveHasError ? 'bg-red-400' : 'bg-slate-500'
            }`}
            animate={{
              scale: effectiveIsLoading || effectiveIsTraining || effectiveIsGenerating ? [1, 1.3, 1] : 1,
            }}
            transition={{ duration: 1, repeat: Infinity }}
          />
          <span className="text-sm text-slate-400">
            {effectiveIsLoading ? '加载中...' :
             effectiveIsTraining ? `训练中 (${effectiveTrainingProgress?.epoch}/${epochs})` :
             effectiveIsGenerating ? '生成中...' :
             effectiveIsReady ? '就绪' :
             effectiveHasError ? '错误' : '空闲'}
          </span>
        </div>
      </div>

      {/* Fallback mode notice */}
      <AnimatePresence>
        {isFallbackMode && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="p-3 bg-amber-500/10 border-b border-amber-500/20"
          >
            <p className="text-amber-400 text-sm flex items-center gap-2">
              <span>⚠️</span>
              {getFallbackStatusMessage()}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading state */}
      <AnimatePresence>
        {effectiveIsLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="p-8 flex flex-col items-center justify-center"
          >
            <LoadingSpinner variant="ring" size="lg" />
            <div className="mt-4 text-center">
              <p className="text-slate-300">{getStageText(stage)}</p>
              <div className="mt-2 w-48 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="mt-1 text-xs text-slate-500">{progress}%</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error state */}
      <AnimatePresence>
        {effectiveHasError && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="p-4 bg-red-500/10 border-b border-red-500/20"
          >
            <p className="text-red-400 text-sm">{error}</p>
            <div className="flex gap-2 mt-2">
              <GlowingButton
                onClick={reset}
                variant="warning"
                size="sm"
              >
                重试
              </GlowingButton>
              <GlowingButton
                onClick={() => {
                  setIsFallbackMode(true);
                  fallbackRef.current = new FallbackTransformer();
                }}
                variant="secondary"
                size="sm"
              >
                使用简化模式
              </GlowingButton>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main content */}
      {!effectiveIsLoading && (
        <div className="p-5 space-y-6">
          {/* Training section */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-slate-400 flex items-center gap-2">
              <span>🎓</span> 训练模型
            </h4>

            <textarea
              value={trainingData}
              onChange={(e) => setTrainingData(e.target.value)}
              placeholder="输入训练数据..."
              className="w-full h-20 px-3 py-2 bg-slate-800/50 border border-slate-600/50 rounded-lg
                text-slate-200 text-sm resize-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50
                transition-all"
            />

            <div className="flex gap-4">
              <div className="flex-1">
                <label className="block text-xs text-slate-500 mb-1">训练轮数</label>
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
                  min={1}
                  max={100}
                  className="w-full px-3 py-1.5 bg-slate-800/50 border border-slate-600/50 rounded-lg
                    text-slate-200 text-sm focus:ring-2 focus:ring-cyan-500/50"
                />
              </div>

              <div className="flex-1">
                <label className="block text-xs text-slate-500 mb-1">学习率</label>
                <input
                  type="number"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
                  min={0.001}
                  max={1}
                  step={0.001}
                  className="w-full px-3 py-1.5 bg-slate-800/50 border border-slate-600/50 rounded-lg
                    text-slate-200 text-sm focus:ring-2 focus:ring-cyan-500/50"
                />
              </div>

              <div className="flex items-end">
                <GlowingButton
                  onClick={handleTrain}
                  disabled={!effectiveIsReady || effectiveIsTraining}
                  loading={effectiveIsTraining}
                  variant="success"
                >
                  开始训练
                </GlowingButton>
              </div>
            </div>

            {/* Training progress */}
            <AnimatePresence>
              {effectiveTrainingProgress && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="p-3 bg-slate-800/30 rounded-lg"
                >
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400">Epoch {effectiveTrainingProgress.epoch}/{epochs}</span>
                    <span className="text-cyan-400">Loss: {effectiveTrainingProgress.loss.toFixed(4)}</span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-green-500 to-emerald-400"
                      initial={{ width: 0 }}
                      animate={{ width: `${(effectiveTrainingProgress.epoch / epochs) * 100}%` }}
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Generation section */}
          <div className="space-y-3 pt-4 border-t border-slate-700/50">
            <h4 className="text-sm font-medium text-slate-400 flex items-center gap-2">
              <span>✨</span> 文本生成
            </h4>

            <div className="flex gap-3">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="输入提示文本..."
                className="flex-1 px-3 py-2 bg-slate-800/50 border border-slate-600/50 rounded-lg
                  text-slate-200 text-sm focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50
                  transition-all"
              />
              <GlowingButton
                onClick={handleGenerate}
                disabled={!effectiveIsReady || effectiveIsGenerating}
                loading={effectiveIsGenerating}
              >
                生成
              </GlowingButton>
            </div>

            {/* Output */}
            <AnimatePresence>
              {outputText && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="p-4 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-lg
                    border border-cyan-500/20"
                >
                  <p className="text-sm text-slate-400 mb-1">生成结果:</p>
                  <p className="text-slate-200 font-medium">{outputText}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      )}
    </div>
  );
}
