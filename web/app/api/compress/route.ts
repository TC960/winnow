import { NextRequest, NextResponse } from "next/server";

// Thin proxy to the FastAPI/Modal compression worker. Keeps the Modal app and
// its API key off the browser, and avoids CORS by going through same-origin.
// Forwards the full merge-aware payload — when `question` is set, the backend
// runs LLMLingua-2 AND AttentionRAG in parallel and merges via `mode`.

const BACKEND = process.env.COMPRESS_BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const hasQuestion = typeof body.question === "string" && body.question.trim().length > 0;
  try {
    const r = await fetch(`${BACKEND}/compress`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        text: body.text,
        rate: body.rate ?? 0.5,
        return_labels: body.return_labels ?? true,
        ...(hasQuestion ? {
          question: body.question.trim(),
          mode: body.mode ?? "intersection",
          chunk_size: body.chunk_size ?? 300,
          top_k: body.top_k ?? 12,
          use_openai_hint: body.use_openai_hint ?? false,
        } : {}),
      }),
    });
    const text = await r.text();
    return new NextResponse(text, {
      status: r.status,
      headers: { "content-type": r.headers.get("content-type") ?? "application/json" },
    });
  } catch (e: any) {
    return NextResponse.json({ error: `backend unreachable: ${e.message}` }, { status: 502 });
  }
}
