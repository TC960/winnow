"use client";

import * as Slider from "@radix-ui/react-slider";
import { useStore } from "@/lib/store";

// Live compression rate slider. Affects the NEXT utterance, not retroactively.
// 0.3 = keep ~30% of tokens (aggressive). 0.8 = light pruning. 0.5 = default.

export function RateSlider() {
  const rate = useStore((s) => s.rate);
  const setRate = useStore((s) => s.setRate);
  return (
    <div className="flex items-center gap-3 min-w-[180px]">
      <span className="text-[10px] font-mono uppercase tracking-wider text-ink-faint">rate</span>
      <Slider.Root
        className="relative flex items-center select-none touch-none flex-1 h-5"
        value={[rate]}
        min={0.25}
        max={0.85}
        step={0.05}
        onValueChange={(v) => setRate(v[0])}
      >
        <Slider.Track className="bg-white/10 relative grow rounded-full h-[3px]">
          <Slider.Range className="absolute h-full rounded-full bg-gradient-to-r from-raw to-keep" />
        </Slider.Track>
        <Slider.Thumb
          className="block w-3.5 h-3.5 bg-ink rounded-full shadow-lg outline-none ring-2 ring-keep/30 hover:ring-keep/60 transition-all"
          aria-label="Compression rate"
        />
      </Slider.Root>
      <span className="text-sm font-mono tabular-nums text-ink w-10 text-right">{rate.toFixed(2)}</span>
    </div>
  );
}
