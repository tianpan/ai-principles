import type { ReactNode } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';

interface AnimatedCardProps {
  children: ReactNode;
  title?: string;
  icon?: ReactNode;
  className?: string;
  glowColor?: string;
  delay?: number;
}

export default function AnimatedCard({
  children,
  title,
  icon,
  className = '',
  glowColor = 'rgba(6, 182, 212, 0.3)',
  delay = 0,
}: AnimatedCardProps) {
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const mouseXSpring = useSpring(x, { stiffness: 500, damping: 100 });
  const mouseYSpring = useSpring(y, { stiffness: 500, damping: 100 });

  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ['7.5deg', '-7.5deg']);
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ['-7.5deg', '7.5deg']);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const xPct = mouseX / width - 0.5;
    const yPct = mouseY / height - 0.5;
    x.set(xPct);
    y.set(yPct);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        rotateX,
        rotateY,
        transformStyle: 'preserve-3d',
      }}
      className={`relative group ${className}`}
    >
      {/* Glow effect */}
      <motion.div
        className="absolute -inset-1 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"
        style={{
          background: `radial-gradient(800px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), ${glowColor}, transparent 40%)`,
        }}
      />

      {/* Card content */}
      <div
        className="relative bg-slate-900/90 backdrop-blur-xl rounded-xl border border-slate-700/50
          shadow-xl shadow-slate-900/50 overflow-hidden"
        style={{ transform: 'translateZ(50px)' }}
      >
        {/* Top gradient line */}
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-500 to-transparent" />

        {/* Header */}
        {(title || icon) && (
          <div className="flex items-center gap-3 px-5 py-4 border-b border-slate-700/50">
            {icon && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: delay + 0.2, type: 'spring' }}
                className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/20
                  flex items-center justify-center text-cyan-400"
              >
                {icon}
              </motion.div>
            )}
            {title && (
              <motion.h3
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: delay + 0.3 }}
                className="text-lg font-semibold text-slate-100"
              >
                {title}
              </motion.h3>
            )}
          </div>
        )}

        {/* Body */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay + 0.4 }}
          className="p-5"
        >
          {children}
        </motion.div>

        {/* Corner accent */}
        <div className="absolute bottom-0 right-0 w-20 h-20 pointer-events-none">
          <div className="absolute bottom-0 right-0 w-full h-full bg-gradient-to-tl from-cyan-500/10 to-transparent" />
        </div>
      </div>
    </motion.div>
  );
}
