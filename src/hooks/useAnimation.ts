import { useState, useEffect, useCallback, useRef } from 'react';
import { useAnimation, AnimationControls, useInView } from 'framer-motion';

interface UseAnimationSequenceOptions {
  autoPlay?: boolean;
  loop?: boolean;
  delay?: number;
  duration?: number;
}

interface AnimationSequenceStep {
  target: string;
  animation: Record<string, unknown>;
}

export function useAnimationSequence(
  steps: AnimationSequenceStep[],
  options: UseAnimationSequenceOptions = {}
) {
  const { autoPlay = true, loop = false, delay = 0, duration = 500 } = options;
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const controls = useAnimation();

  useEffect(() => {
    if (!isPlaying || steps.length === 0) return;

    const timeout = setTimeout(() => {
      const step = steps[currentStep];
      if (step) {
        controls.start(step.animation as Parameters<typeof controls.start>[0]);
      }

      setCurrentStep((prev) => {
        const next = prev + 1;
        if (next >= steps.length) {
          if (loop) return 0;
          setIsPlaying(false);
          return prev;
        }
        return next;
      });
    }, delay + duration);

    return () => clearTimeout(timeout);
  }, [currentStep, isPlaying, steps, controls, delay, duration, loop]);

  const play = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(true);
  }, []);

  const pause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const reset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
    controls.stop();
  }, [controls]);

  return { controls, currentStep, isPlaying, play, pause, reset };
}

export function useScrollAnimation(threshold = 0.1) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, amount: threshold });
  const controls = useAnimation();

  useEffect(() => {
    if (isInView) {
      controls.start('visible');
    }
  }, [isInView, controls]);

  return { ref, controls, isInView };
}

export function useStaggerAnimation(itemCount: number, staggerDelay = 0.1) {
  const controls = useAnimation();
  const [isComplete, setIsComplete] = useState(false);

  const start = useCallback(async () => {
    for (let i = 0; i < itemCount; i++) {
      await controls.start((i) => ({
        opacity: 1,
        y: 0,
        transition: { delay: i * staggerDelay },
      }));
    }
    setIsComplete(true);
  }, [controls, itemCount, staggerDelay]);

  return { controls, start, isComplete };
}

export function useReducedMotion() {
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mediaQuery.matches);

    const handler = (event: MediaQueryListEvent) => {
      setReducedMotion(event.matches);
    };

    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  return reducedMotion;
}

export function useTypewriter(text: string, speed = 50, delay = 0) {
  const [displayText, setDisplayText] = useState('');
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    setDisplayText('');
    setIsComplete(false);

    const startTimeout = setTimeout(() => {
      let index = 0;
      const interval = setInterval(() => {
        if (index < text.length) {
          setDisplayText(text.slice(0, index + 1));
          index++;
        } else {
          clearInterval(interval);
          setIsComplete(true);
        }
      }, speed);

      return () => clearInterval(interval);
    }, delay);

    return () => clearTimeout(startTimeout);
  }, [text, speed, delay]);

  return { displayText, isComplete };
}

export function useCountAnimation(
  end: number,
  duration = 1000,
  startOnMount = true
) {
  const [count, setCount] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const animate = useCallback(() => {
    setIsAnimating(true);
    const startTime = Date.now();

    const update = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Easing function (ease-out)
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(Math.floor(eased * end));

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        setIsAnimating(false);
      }
    };

    requestAnimationFrame(update);
  }, [end, duration]);

  useEffect(() => {
    if (startOnMount) {
      animate();
    }
  }, [startOnMount, animate]);

  return { count, isAnimating, animate };
}
