// src/components/hero/TokenFlow.tsx
import { motion } from 'framer-motion';

const tokens = ['T', 'r', 'a', 'n', 's', 'f', 'o', 'r', 'm', 'e', 'r'];

interface TokenFlowProps {
  isActive?: boolean;
}

export default function TokenFlow({ isActive = true }: TokenFlowProps) {
  return (
    <div className="flex items-center justify-center gap-2 my-8">
      {tokens.map((token, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, scale: 0, y: -50 }}
          animate={{
            opacity: 1,
            scale: isActive ? 1 : 0.9,
            y: 0
          }}
          transition={{
            duration: 0.5,
            delay: i * 0.1,
            repeat: isActive ? Infinity : 0,
            repeatDelay: 3,
          }}
          className="relative"
        >
          {/* Token 方块 */}
          <motion.div
            className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold text-xl shadow-lg"
            animate={{
              boxShadow: isActive
                ? ['0 10px 30px rgba(6, 182, 212, 0.3)', '0 10px 50px rgba(6, 182, 212, 0.5)', '0 10px 30px rgba(6, 182, 212, 0.3)']
                : '0 5px 15px rgba(6, 182, 212, 0.2)',
            }}
            transition={{
              duration: 2,
              repeat: isActive ? Infinity : 0,
              delay: i * 0.1,
            }}
          >
            {token}
          </motion.div>

          {/* 发光效果 */}
          <motion.div
            className="absolute inset-0 rounded-lg bg-cyan-400 blur-xl opacity-30"
            animate={{
              opacity: isActive ? [0.2, 0.5, 0.2] : 0.1,
              scale: isActive ? [1, 1.2, 1] : 1,
            }}
            transition={{
              duration: 2,
              repeat: isActive ? Infinity : 0,
              delay: i * 0.1,
            }}
          />

          {/* 连接线 */}
          {i < tokens.length - 1 && (
            <motion.div
              className="absolute top-1/2 left-full w-2 h-0.5"
              style={{ backgroundColor: isActive ? '#22d3ee' : '#475569' }}
              initial={{ scaleX: 0 }}
              animate={{ scaleX: isActive ? 1 : 0.5 }}
              transition={{
                duration: 0.3,
                delay: i * 0.1 + 0.3,
                repeat: isActive ? Infinity : 0,
                repeatDelay: 3,
              }}
              style={{ transformOrigin: 'left' }}
            />
          )}
        </motion.div>
      ))}
    </div>
  );
}
