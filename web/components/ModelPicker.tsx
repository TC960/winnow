"use client";

import { ChevronDown } from "lucide-react";
import { MODELS, type ModelId } from "@/lib/tokens";
import { useStore } from "@/lib/store";

export function ModelPicker() {
  const model = useStore((s) => s.model);
  const setModel = useStore((s) => s.setModel);
  return (
    <label className="relative flex items-center gap-2">
      <span className="text-[10px] font-mono uppercase tracking-wider text-ink-faint">model</span>
      <div className="relative">
        <select
          value={model}
          onChange={(e) => setModel(e.target.value as ModelId)}
          className="appearance-none glass rounded-full pl-3 pr-8 py-1.5 text-[12px] text-ink focus:outline-none cursor-pointer"
        >
          {Object.values(MODELS).map((m) => (
            <option key={m.id} value={m.id}>
              {m.label} — ${m.inputPerMTok}/M
            </option>
          ))}
        </select>
        <ChevronDown className="w-3 h-3 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-ink-dim" />
      </div>
    </label>
  );
}
