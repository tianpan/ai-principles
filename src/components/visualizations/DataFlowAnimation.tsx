import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

interface DataFlowAnimationProps {
  stages: string[];
  activeStage?: number;
  onComplete?: () => void;
  loop?: boolean;
  speed?: number;
}

export default function DataFlowAnimation({
  stages,
  activeStage = 0,
  onComplete,
  loop = false,
  speed = 2000,
}: DataFlowAnimationProps) {
  const [currentStage, setCurrentStage] = useState(activeStage);
  const [particles, setParticles] = useState<{ id: number; x: number; progress: number }[]>([]);
  const particleIdRef = useRef(0);

  // Animate through stages
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStage((prev) => {
        const next = prev + 1;
        if (next >= stages.length) {
          if (loop) {
            return 0;
          }
          onComplete?.();
          return prev;
        }
        return next;
      });
    }, speed);

    return () => clearInterval(interval);
  }, [stages.length, loop, speed, onComplete]);

  // Generate particles flowing between stages
  useEffect(() => {
    const particleInterval = setInterval(() => {
      const stageWidth = 100 / stages.length;
      const startX = (currentStage + 0.5) * stageWidth;
      const endX = ((currentStage + 1) % stages.length + 0.5) * stageWidth;

      setParticles((prev) => [
        ...prev.slice(-10), // Keep only last 10 particles
        { id: particleIdRef.current++, x: startX, progress: 0 },
      ]);
    }, 200);

    return () => clearInterval(particleInterval);
  }, [currentStage, stages.length]);

  // Animate particles
  useEffect(() => {
    const animationFrame = requestAnimationFrame(function animate() {
      setParticles((prev) =>
        prev
          .map((p) => ({ ...p, progress: p.progress + 0.02 }))
          .filter((p) => p.progress < 1)
      );
      requestAnimationFrame(animate);
    });

    return () => cancelAnimationFrame(animationFrame);
  }, []);

  return (
    <div className="relative w-full h-40 overflow-hidden">
      {/* Stage nodes */}
      <div className="absolute inset-0 flex items-center justify-between px-8">
        {stages.map((stage, i) => (
          <motion.div
            key={stage}
            initial={{ scale: 0.8, opacity: 0.5 }}
            animate={{
              scale: i === currentStage ? 1.2 : 0.8,
              opacity: i === currentStage ? 1 : 0.5,
              boxShadow:
                i === currentStage
                  ? '0 0 30px rgba(6, 182, 212, 0.5)'
                  : '0 0 10px rgba(6, 182, 212, 0.2)',
            }}
            transition={{ duration: 0.3 }}
            className="relative z-10"
          >
            <div
              className={`w-20 h-20 rounded-xl flex items-center justify-center text-sm font-medium
                ${i === currentStage
                  ? 'bg-gradient-to-br from-cyan-500 to-blue-600 text-white'
                  : 'bg-slate-800 text-slate-400 border border-slate-700'
                }`}
            >
              {stage}
            </div>
            {/* Label */}
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: i === currentStage ? 1 : 0.5, y: 0 }}
              className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-xs text-slate-500 whitespace-nowrap"
            >
              Step {i + 1}
            </motion.div>
          </motion.div>
        ))}
      </div>

      {/* Connection lines */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        {stages.slice(0, -1).map((_, i) => {
          const startX = ((i + 0.5) / stages.length) * 100;
          const endX = ((i + 1.5) / stages.length) * 100;
          const isActive = i === currentStage || i === currentStage - 1;

          return (
            <motion.line
              key={i}
              x1={`${startX}%`}
              y1="50%"
              x2={`${endX}%`}
              y2="50%"
              stroke={isActive ? '#06b6d4' : '#334155'}
              strokeWidth={isActive ? 3 : 1}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.5 }}
            />
          );
        })}
      </svg>

      {/* Flowing particles */}
      {particles.map((particle) => {
        const stageWidth = 100 / stages.length;
        const startX = particle.x;
        const endX = ((Math.floor(particle.x / stageWidth) + 1.5) % 100) * stageWidth;
        const currentX = startX + (endX - startX) * particle.progress;

        return (
          <motion.div
            key={particle.id}
            className="absolute w-3 h-3 rounded-full bg-cyan-400 shadow-lg shadow-cyan-500/50"
            style={{
              left: `${currentX}%`,
              top: '50%',
              transform: 'translate(-50%, -50%)',
            }}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 - particle.progress }}
          />
        );
      })}

      {/* Progress indicator */}
      <div className="absolute bottom-0 left-0 right-0 h-1 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
          initial={{ width: '0%' }}
          animate={{ width: `${((currentStage + 1) / stages.length) * 100}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>
    </div>
  );
}
