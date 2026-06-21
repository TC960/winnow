"use client";

import { TranscriptColumn } from "./TranscriptColumn";
import { CompressedColumn } from "./CompressedColumn";
import { StatsBar } from "./StatsBar";
import { RateSlider } from "./RateSlider";
import { ModelPicker } from "./ModelPicker";
import { LanguagePicker } from "./LanguagePicker";
import { QABox } from "./QABox";
import { DirectorOverlay } from "./DirectorOverlay";

// The original Compare tab: live raw vs compressed feed, stats bar, A/B Q&A.
// Everything below the tab nav and the global SourceToggle.

export function CompareView() {
  return (
    <>
      <div className="glass rounded-2xl px-5 py-3 flex items-center gap-5 flex-wrap">
        <RateSlider />
        <div className="w-px h-5 bg-white/10" />
        <ModelPicker />
        <div className="w-px h-5 bg-white/10" />
        <LanguagePicker />
        <div className="ml-auto">
          <DirectorOverlay />
        </div>
      </div>

      <StatsBar />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 flex-1 min-h-[420px] h-[55vh]">
        <TranscriptColumn />
        <CompressedColumn />
      </div>

      <QABox />
    </>
  );
}
