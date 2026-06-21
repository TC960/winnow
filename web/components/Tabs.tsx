"use client";

import { motion } from "framer-motion";
import { GitCompare, FlaskConical } from "lucide-react";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";

// Top-level tab nav. Switching tabs DOES NOT reset session state — Compare
// and Test share the same compressed transcript, so flipping between them
// is just a view change.

export function Tabs() {
  const tab = useStore((s) => s.tab);
  const setTab = useStore((s) => s.setTab);

  return (
    <div className="glass rounded-full p-1 flex relative">
      <TabBtn id="compare" active={tab === "compare"} onClick={() => setTab("compare")}>
        <GitCompare className="w-3.5 h-3.5" /> Compare
      </TabBtn>
      <TabBtn id="test" active={tab === "test"} onClick={() => setTab("test")}>
        <FlaskConical className="w-3.5 h-3.5" /> Test
      </TabBtn>
    </div>
  );
}

function TabBtn({
  id,
  active,
  onClick,
  children,
}: {
  id: string;
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "relative px-4 py-1.5 rounded-full text-[12px] font-medium tracking-wide flex items-center gap-1.5 transition-colors z-10",
        active ? "text-ink" : "text-ink-dim hover:text-ink"
      )}
    >
      {active && (
        <motion.span
          layoutId="tab-pill"
          className="absolute inset-0 rounded-full bg-gradient-to-br from-keep/25 to-cyan-accent/15 border border-keep/35 -z-10"
          transition={{ type: "spring", duration: 0.35, bounce: 0.18 }}
        />
      )}
      {children}
    </button>
  );
}
