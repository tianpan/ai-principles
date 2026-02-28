import type { ReactNode } from 'react';
import { useState } from 'react';
import { motion } from 'framer-motion';

interface GlowingButtonProps {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'success' | 'warning';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  icon?: ReactNode;
  className?: string;
}

const variants = {
  primary: {
    gradient: 'from-cyan-500 to-blue-600',
    glow: 'rgba(6, 182, 212, 0.5)',
    shadow: 'shadow-cyan-500/30',
    text: 'text-white',
    border: 'border-cyan-400/30',
  },
  secondary: {
    gradient: 'from-slate-600 to-slate-700',
    glow: 'rgba(100, 116, 139, 0.5)',
    shadow: 'shadow-slate-500/30',
    text: 'text-slate-200',
    border: 'border-slate-500/30',
  },
  success: {
    gradient: 'from-green-500 to-emerald-600',
    glow: 'rgba(16, 185, 129, 0.5)',
    shadow: 'shadow-green-500/30',
    text: 'text-white',
    border: 'border-green-400/30',
  },
  warning: {
    gradient: 'from-amber-500 to-orange-600',
    glow: 'rgba(245, 158, 11, 0.5)',
    shadow: 'shadow-amber-500/30',
    text: 'text-white',
    border: 'border-amber-400/30',
  },
};

const sizes = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-5 py-2.5 text-base',
  lg: 'px-7 py-3.5 text-lg',
};

export default function GlowingButton({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  icon,
  className = '',
}: GlowingButtonProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [isPressed, setIsPressed] = useState(false);

  const style = variants[variant];

  return (
    <motion.button
      onClick={onClick}
      disabled={disabled || loading}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      onMouseDown={() => setIsPressed(true)}
      onMouseUp={() => setIsPressed(false)}
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      className={`relative group ${className}`}
    >
      {/* Glow background */}
      <motion.div
        className={`absolute inset-0 rounded-lg bg-gradient-to-r ${style.gradient} blur-xl`}
        animate={{
          opacity: isHovered && !disabled ? 0.6 : 0,
          scale: isHovered && !disabled ? 1.1 : 1,
        }}
        transition={{ duration: 0.3 }}
      />

      {/* Button content */}
      <div
        className={`relative flex items-center justify-center gap-2 rounded-lg font-medium
          bg-gradient-to-r ${style.gradient} ${style.text} ${sizes[size]}
          border ${style.border} shadow-lg ${style.shadow}
          transition-all duration-300
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          ${isPressed && !disabled ? 'brightness-90' : ''}
        `}
      >
        {/* Shimmer effect */}
        <motion.div
          className="absolute inset-0 rounded-lg overflow-hidden"
          animate={{
            x: isHovered ? ['0%', '200%'] : '-100%',
          }}
          transition={{
            duration: 1.5,
            repeat: isHovered ? Infinity : 0,
            ease: 'linear',
          }}
        >
          <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent w-1/2" />
        </motion.div>

        {/* Icon */}
        {icon && (
          <motion.span
            animate={{ rotate: loading ? 360 : 0 }}
            transition={{ duration: 1, repeat: loading ? Infinity : 0, ease: 'linear' }}
            className="relative"
          >
            {icon}
          </motion.span>
        )}

        {/* Text */}
        <span className="relative">{children}</span>

        {/* Loading spinner */}
        {loading && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center bg-inherit rounded-lg"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <motion.div
              className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
            />
          </motion.div>
        )}
      </div>

      {/* Ripple effect on click */}
      {isPressed && !disabled && (
        <motion.div
          className="absolute inset-0 rounded-lg bg-white/30"
          initial={{ scale: 0, opacity: 1 }}
          animate={{ scale: 1.5, opacity: 0 }}
          transition={{ duration: 0.5 }}
        />
      )}
    </motion.button>
  );
}
