"use client";

import { cn } from "@/lib/cn";

// Renders LLMLingua-2 word-level keep/drop labels with kept words ink-colored
// and dropped words struck-through. Falls back to plain text if no labels.

export function StrikeText({
  labels,
  fallback,
}: {
  labels?: [string, number][] | null;
  fallback?: string;
}) {
  const hasLabels = Array.isArray(labels) && labels.length > 0 && Array.isArray(labels[0]);
  if (!hasLabels) {
    return <span className="text-ink">{fallback ?? ""}</span>;
  }
  return (
    <span>
      {(labels as [string, number][]).map(([word, keep], i, arr) => (
        <span
          key={i}
          className={cn(
            keep
              ? "text-ink"
              : "text-ink-faint line-through decoration-raw/60 decoration-[1.5px]"
          )}
        >
          {word}
          {i < arr.length - 1 ? " " : ""}
        </span>
      ))}
    </span>
  );
}
