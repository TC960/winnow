"use client";

import { ChevronDown } from "lucide-react";
import { useStore } from "@/lib/store";

const LANGUAGES = [
  { value: "en-US", label: "English (US)" },
  { value: "en-GB", label: "English (UK)" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "hi", label: "Hindi" },
  { value: "ja", label: "Japanese" },
  { value: "multi", label: "Multilingual (mixed)" },
];

export function LanguagePicker() {
  const language = useStore((s) => s.language);
  const setLanguage = useStore((s) => s.setLanguage);
  return (
    <label className="relative flex items-center gap-2">
      <span className="text-[10px] font-mono uppercase tracking-wider text-ink-faint">lang</span>
      <div className="relative">
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="appearance-none glass rounded-full pl-3 pr-8 py-1.5 text-[12px] text-ink focus:outline-none cursor-pointer"
        >
          {LANGUAGES.map((l) => (
            <option key={l.value} value={l.value}>{l.label}</option>
          ))}
        </select>
        <ChevronDown className="w-3 h-3 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-ink-dim" />
      </div>
    </label>
  );
}
