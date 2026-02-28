// src/components/hero/ParticleBg.tsx
import { useEffect, useRef } from 'react';

type AnimationPhase = 'token-flow' | 'attention' | 'combined';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  hue: number;
}

interface ParticleBgProps {
  activePhase?: AnimationPhase;
}

// Different color schemes per phase
const phaseColors: Record<AnimationPhase, { primary: number; secondary: number }> = {
  'token-flow': { primary: 190, secondary: 210 }, // Cyan to Blue
  'attention': { primary: 280, secondary: 320 }, // Purple to Pink
  'combined': { primary: 160, secondary: 190 }, // Green to Cyan
};

export default function ParticleBg({ activePhase = 'combined' }: ParticleBgProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>();
  const phaseRef = useRef<AnimationPhase>(activePhase);

  // Update phase ref when prop changes
  useEffect(() => {
    phaseRef.current = activePhase;
  }, [activePhase]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 设置画布大小
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // 初始化粒子
    const particleCount = 150;
    particlesRef.current = Array.from({ length: particleCount }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 2 + 1,
      opacity: Math.random() * 0.5 + 0.2,
      hue: Math.random() * 30,
    }));

    // 动画循环
    const animate = () => {
      const currentPhase = phaseRef.current;
      const colors = phaseColors[currentPhase];

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 绘制粒子
      particlesRef.current.forEach((p) => {
        // 更新位置
        p.x += p.vx;
        p.y += p.vy;

        // 边界检查
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        // 绘制 with phase-based color
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        const hue = colors.primary + (p.hue / 30) * (colors.secondary - colors.primary);
        ctx.fillStyle = `hsla(${hue}, 80%, 60%, ${p.opacity})`;
        ctx.fill();
      });

      // 绘制连线
      particlesRef.current.forEach((p1, i) => {
        particlesRef.current.slice(i + 1).forEach((p2) => {
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 100) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            const hue = colors.primary;
            ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${0.1 * (1 - dist / 100)})`;
            ctx.stroke();
          }
        });
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none transition-opacity duration-1000"
      style={{ zIndex: -1 }}
    />
  );
}
