"use client";

import { useRef, useState } from "react";
import { Plus, X, Mic, FileText, Upload } from "lucide-react";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/cn";

// Sources panel. ALWAYS shows the live "Compressed transcript" virtual source
// at the top — it's piped in automatically from the Compare-tab session.
// Below that, user-added sources via either file upload (text/markdown/code)
// or paste.

// Files we can read directly in the browser as plain text.
const TEXT_ACCEPT = ".txt,.md,.markdown,.json,.csv,.tsv,.log,.xml,.html,.htm,.yaml,.yml,.py,.js,.ts,.tsx,.jsx,.go,.rs,.java,.c,.cpp,.h,.css,.sh,text/*";

export function SourcesPanel() {
  const rows = useStore((s) => s.rows);
  const extras = useStore((s) => s.extraSources);
  const addSource = useStore((s) => s.addSource);
  const removeSource = useStore((s) => s.removeSource);
  const [adding, setAdding] = useState(false);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const compressedText = rows
    .filter((r) => r.compressed)
    .map((r) => r.compressed!.text)
    .join(" ");
  const compressedTokens = rows
    .filter((r) => r.compressed)
    .reduce((a, r) => a + (r.compressed?.compressedTokens ?? 0), 0);

  function submit() {
    if (!title.trim() || !content.trim()) return;
    addSource({ title: title.trim(), content: content.trim() });
    setTitle("");
    setContent("");
    setAdding(false);
  }

  async function onFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setUploadErr(null);
    const failures: string[] = [];
    for (const file of Array.from(files)) {
      try {
        const text = await file.text();
        if (!text.trim()) {
          failures.push(`${file.name} (empty)`);
          continue;
        }
        addSource({ title: file.name, content: text });
      } catch {
        failures.push(file.name);
      }
    }
    if (failures.length) {
      setUploadErr(`Couldn't read: ${failures.join(", ")}. Use a text-based file (.txt, .md, .json, code).`);
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  return (
    <section className="glass rounded-2xl flex flex-col h-full overflow-hidden">
      <header className="flex items-center justify-between px-5 py-3 border-b border-white/5">
        <h2 className="text-sm font-semibold tracking-wide">SOURCES</h2>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="text-[11px] font-mono uppercase tracking-wider px-2 py-1 rounded-full bg-white/5 hover:bg-white/10 text-ink-dim hover:text-ink transition flex items-center gap-1.5"
            title="Upload text files"
          >
            <Upload className="w-3 h-3" /> upload
          </button>
          <button
            onClick={() => setAdding((a) => !a)}
            className="text-[11px] font-mono uppercase tracking-wider px-2 py-1 rounded-full bg-white/5 hover:bg-white/10 text-ink-dim hover:text-ink transition flex items-center gap-1.5"
            title="Paste text"
          >
            <Plus className="w-3 h-3" /> paste
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={TEXT_ACCEPT}
            onChange={(e) => onFiles(e.target.files)}
            className="hidden"
          />
        </div>
      </header>

      <div className="flex-1 overflow-y-auto scroll-soft p-3 space-y-2">
        {/* Pinned virtual source: the live compressed transcript. */}
        <div className="rounded-xl border border-keep/30 bg-keep/5 p-3">
          <div className="flex items-start gap-2.5">
            <Mic className="w-3.5 h-3.5 text-keep mt-0.5" />
            <div className="flex-1 min-w-0">
              <div className="text-[13px] font-semibold text-ink truncate">Compressed transcript</div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">
                {rows.length} utt · {compressedTokens.toLocaleString()} tokens · live
              </div>
              <div className="text-[12px] text-ink-dim mt-1.5 line-clamp-2">
                {compressedText || "Run a session on the Compare tab to feed this in."}
              </div>
            </div>
          </div>
        </div>

        {/* User-added sources */}
        {extras.map((s) => (
          <div key={s.id} className="rounded-xl border border-white/10 bg-white/3 p-3 group">
            <div className="flex items-start gap-2.5">
              <FileText className="w-3.5 h-3.5 text-cyan-accent mt-0.5" />
              <div className="flex-1 min-w-0">
                <div className="text-[13px] font-semibold text-ink truncate">{s.title}</div>
                <div className="text-[10px] font-mono uppercase tracking-wider text-ink-faint mt-0.5">
                  {s.content.length.toLocaleString()} chars
                </div>
                <div className="text-[12px] text-ink-dim mt-1.5 line-clamp-2">{s.content}</div>
              </div>
              <button
                onClick={() => removeSource(s.id)}
                className="opacity-0 group-hover:opacity-100 text-ink-faint hover:text-raw transition"
                title="Remove source"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        ))}

        {uploadErr && (
          <div className="text-[11px] text-raw bg-raw/5 border border-raw/20 rounded-lg px-3 py-2">
            {uploadErr}
          </div>
        )}

        {/* Paste form */}
        {adding && (
          <div className="rounded-xl border border-cyan-accent/30 bg-cyan-accent/5 p-3 space-y-2">
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Source title (e.g. 'Last week's notes')"
              className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-[13px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-cyan-accent/50"
            />
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Paste content here…"
              rows={4}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-[13px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-cyan-accent/50 resize-none"
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => { setAdding(false); setTitle(""); setContent(""); }}
                className="text-[11px] font-mono uppercase tracking-wider px-3 py-1 rounded-full text-ink-dim hover:text-ink"
              >
                cancel
              </button>
              <button
                onClick={submit}
                disabled={!title.trim() || !content.trim()}
                className="text-[11px] font-mono uppercase tracking-wider px-3 py-1 rounded-full bg-keep/15 border border-keep/40 text-keep hover:bg-keep/25 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                add
              </button>
            </div>
          </div>
        )}

        {!adding && extras.length === 0 && (
          <div className={cn("text-[12px] text-ink-faint italic px-2 pt-2")}>
            Upload or paste documents to study alongside your transcript.
          </div>
        )}
      </div>
    </section>
  );
}
