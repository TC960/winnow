"use client";

import { useEffect, useRef, useState } from "react";

// Smoothly tweens a displayed number toward the target. We use this for token
// counts and savings so the stats bar never *jumps* — judges feel the motion.

export function AnimatedNumber({
  value,
  format = (n) => n.toLocaleString(),
  duration = 350,
  className,
}: {
  value: number;
  format?: (n: number) => string;
  duration?: number;
  className?: string;
}) {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  const startRef = useRef<number | null>(null);
  const targetRef = useRef(value);

  useEffect(() => {
    fromRef.current = display;
    targetRef.current = value;
    startRef.current = null;
    let raf = 0;
    const step = (t: number) => {
      if (startRef.current === null) startRef.current = t;
      const k = Math.min(1, (t - startRef.current) / duration);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - k, 3);
      const next = fromRef.current + (targetRef.current - fromRef.current) * eased;
      setDisplay(next);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, duration]);

  return <span className={className}>{format(display)}</span>;
}
