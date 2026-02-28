import { motion } from 'framer-motion';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'dots' | 'ring' | 'pulse' | 'bars';
  color?: string;
  text?: string;
}

const sizes = {
  sm: { container: 'w-8 h-8', element: 'w-1.5 h-1.5' },
  md: { container: 'w-12 h-12', element: 'w-2 h-2' },
  lg: { container: 'w-16 h-16', element: 'w-3 h-3' },
};

export default function LoadingSpinner({
  size = 'md',
  variant = 'dots',
  color = '#06b6d4',
  text,
}: LoadingSpinnerProps) {
  const sizeConfig = sizes[size];

  const containerVariants = {
    animate: {
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const dotVariants = {
    initial: { scale: 0, opacity: 0 },
    animate: {
      scale: [0, 1, 0],
      opacity: [0, 1, 0],
      transition: {
        duration: 1,
        repeat: Infinity,
        ease: 'easeInOut',
      },
    },
  };

  const barVariants = {
    initial: { scaleY: 0.3 },
    animate: (i: number) => ({
      scaleY: [0.3, 1, 0.3],
      transition: {
        duration: 0.8,
        repeat: Infinity,
        delay: i * 0.1,
        ease: 'easeInOut',
      },
    }),
  };

  const renderSpinner = () => {
    switch (variant) {
      case 'dots':
        return (
          <motion.div
            className={`${sizeConfig.container} flex items-center justify-center gap-1`}
            variants={containerVariants}
            initial="initial"
            animate="animate"
          >
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                className={`${sizeConfig.element} rounded-full`}
                style={{ backgroundColor: color }}
                variants={dotVariants}
              />
            ))}
          </motion.div>
        );

      case 'ring':
        return (
          <div className={`${sizeConfig.container} relative`}>
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-transparent"
              style={{ borderTopColor: color }}
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            />
            <motion.div
              className="absolute inset-1 rounded-full border-2 border-transparent"
              style={{ borderTopColor: color, opacity: 0.5 }}
              animate={{ rotate: -360 }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
            />
          </div>
        );

      case 'pulse':
        return (
          <div className={`${sizeConfig.container} relative`}>
            <motion.div
              className="absolute inset-0 rounded-full"
              style={{ backgroundColor: color }}
              animate={{
                scale: [1, 1.5, 1],
                opacity: [0.5, 0, 0.5],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            />
            <motion.div
              className="absolute inset-2 rounded-full"
              style={{ backgroundColor: color }}
              animate={{
                scale: [1, 0.8, 1],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            />
          </div>
        );

      case 'bars':
        return (
          <div className={`${sizeConfig.container} flex items-end justify-center gap-0.5`}>
            {[0, 1, 2, 3, 4].map((i) => (
              <motion.div
                key={i}
                className={`${sizeConfig.element} rounded-sm`}
                style={{ backgroundColor: color }}
                custom={i}
                variants={barVariants}
                initial="initial"
                animate="animate"
              />
            ))}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col items-center justify-center gap-3">
      {renderSpinner()}
      {text && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-sm text-slate-400"
        >
          {text}
        </motion.p>
      )}
    </div>
  );
}
