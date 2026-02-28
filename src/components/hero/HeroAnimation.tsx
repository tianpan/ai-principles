// src/components/hero/HeroAnimation.tsx
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ParticleBg from './ParticleBg';
import TokenFlow from './TokenFlow';
import AttentionViz from './AttentionViz';

type AnimationPhase = 'token-flow' | 'attention' | 'combined';

const PHASE_DURATION = 10000; // 10 seconds per phase
const PHASES: AnimationPhase[] = ['token-flow', 'attention', 'combined'];

const phaseLabels: Record<AnimationPhase, string> = {
  'token-flow': 'Token 嵌入',
  'attention': 'Self-Attention',
  'combined': '完整流程',
};

export default function HeroAnimation() {
  const [currentPhaseIndex, setCurrentPhaseIndex] = useState(0);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  const currentPhase = PHASES[currentPhaseIndex];

  // Check for reduced motion preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => setPrefersReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // Animation loop - cycle through phases every 10 seconds
  useEffect(() => {
    if (prefersReducedMotion) return;

    const interval = setInterval(() => {
      setCurrentPhaseIndex((prev) => (prev + 1) % PHASES.length);
    }, PHASE_DURATION);

    return () => clearInterval(interval);
  }, [prefersReducedMotion]);

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
      {/* 粒子背景 */}
      <ParticleBg activePhase={currentPhase} />

      {/* 主内容 */}
      <div className="relative z-10 text-center px-4">
        {/* 标题 */}
        <motion.h1
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-5xl md:text-7xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent"
        >
          Attention Is All You Need
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="text-xl text-slate-400 mb-8"
        >
          从零开始理解 Transformer
        </motion.p>

        {/* Phase indicator */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentPhase}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.3 }}
            className="mb-4"
          >
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-800/50 border border-slate-700/50">
              <motion.div
                className="w-2 h-2 rounded-full bg-cyan-400"
                animate={{
                  scale: [1, 1.3, 1],
                  opacity: [0.7, 1, 0.7],
                }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
              <span className="text-sm text-slate-400">
                {phaseLabels[currentPhase]}
              </span>
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Progress bar for phase */}
        <div className="w-48 mx-auto mb-6 h-0.5 bg-slate-800 rounded-full overflow-hidden">
          <motion.div
            key={currentPhase}
            className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
            initial={{ width: '0%' }}
            animate={{ width: '100%' }}
            transition={{ duration: PHASE_DURATION / 1000, ease: 'linear' }}
          />
        </div>

        {/* Token 流动 */}
        <motion.div
          animate={{
            opacity: currentPhase === 'attention' ? 0.3 : 1,
            scale: currentPhase === 'token-flow' ? 1.05 : 1,
          }}
          transition={{ duration: 0.5 }}
        >
          <TokenFlow isActive={currentPhase !== 'attention'} />
        </motion.div>

        {/* Attention 可视化 */}
        <motion.div
          className="max-w-2xl mx-auto"
          animate={{
            opacity: currentPhase === 'token-flow' ? 0.3 : 1,
            scale: currentPhase === 'attention' ? 1.05 : 1,
          }}
          transition={{ duration: 0.5 }}
        >
          <AttentionViz isActive={currentPhase !== 'token-flow'} />
        </motion.div>

        {/* CTA 按钮 */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5, duration: 0.5 }}
          className="flex gap-4 justify-center mt-8"
        >
          <a
            href="/prerequisites"
            className="px-8 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-medium hover:shadow-lg hover:shadow-cyan-500/30 transition-all"
          >
            开始学习
          </a>
          <a
            href="/lab/intro"
            className="px-8 py-3 rounded-lg border border-cyan-500 text-cyan-400 font-medium hover:bg-cyan-500/10 transition-all"
          >
            进入实验室
          </a>
        </motion.div>

        {/* Phase dots indicator */}
        <div className="flex justify-center gap-2 mt-6">
          {PHASES.map((phase, index) => (
            <button
              key={phase}
              onClick={() => setCurrentPhaseIndex(index)}
              className={`w-2 h-2 rounded-full transition-all ${
                index === currentPhaseIndex
                  ? 'bg-cyan-400 w-6'
                  : 'bg-slate-600 hover:bg-slate-500'
              }`}
              aria-label={`切换到 ${phaseLabels[phase]}`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
