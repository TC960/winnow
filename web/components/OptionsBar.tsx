"use client";

import * as Switch from "@radix-ui/react-switch";
import { useStore } from "@/lib/store";
import { Users, Filter } from "lucide-react";

// Smaller secondary toggles row: diarization + filler prefilter.
// Both feed the SAME pipeline — diarize affects Deepgram options, fillers
// affect the text we send to LLMLingua.

export function OptionsBar() {
  const diarize = useStore((s) => s.diarize);
  const setDiarize = useStore((s) => s.setDiarize);
  const stripF = useStore((s) => s.stripFillersFirst);
  const setStripF = useStore((s) => s.setStripFillers);
  return (
    <div className="flex items-center gap-5">
      <SwitchRow
        checked={diarize}
        onCheckedChange={setDiarize}
        icon={<Users className="w-3.5 h-3.5" />}
        label="diarize"
      />
      <SwitchRow
        checked={stripF}
        onCheckedChange={setStripF}
        icon={<Filter className="w-3.5 h-3.5" />}
        label="strip fillers first"
      />
    </div>
  );
}

function SwitchRow({
  checked,
  onCheckedChange,
  icon,
  label,
}: {
  checked: boolean;
  onCheckedChange: (b: boolean) => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer">
      <Switch.Root
        checked={checked}
        onCheckedChange={onCheckedChange}
        className="w-8 h-[18px] bg-white/10 rounded-full relative data-[state=checked]:bg-keep/50 transition-colors"
      >
        <Switch.Thumb className="block w-3.5 h-3.5 bg-ink rounded-full transition-transform translate-x-0.5 data-[state=checked]:translate-x-[14px]" />
      </Switch.Root>
      <span className="flex items-center gap-1.5 text-[11px] text-ink-dim font-mono uppercase tracking-wider">
        {icon}
        {label}
      </span>
    </label>
  );
}
