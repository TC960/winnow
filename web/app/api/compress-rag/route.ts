import { NextRequest, NextResponse } from "next/server";

// Proxy to FastAPI /compress_rag. Two-stage, question-aware: BGE reranker picks
// the documents most relevant to `question`, then LLMLingua-2 token-compresses
// the survivors at `rate`. Returns the assembled (instruction + context + question)
// prompt plus token counts and coarse-stage diagnostics.

const BACKEND = process.env.COMPRESS_BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.json();
  try {
    const r = await fetch(`${BACKEND}/compress_rag`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        instruction: body.instruction ?? "",
        question: body.question ?? "",
        documents: body.documents ?? [],
        rate: body.rate ?? 0.5,
        target_token: body.target_token ?? -1,
        top_k: body.top_k ?? null,
        score_threshold: body.score_threshold ?? null,
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
