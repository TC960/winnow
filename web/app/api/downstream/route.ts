import { NextRequest, NextResponse } from "next/server";

// Proxy to downstream.py (provider-agnostic blackbox LLM API on :8100).
// Keeps provider API keys server-side. Used by the Test-tab SpeakCard to send
// the LLMLingua-compressed prompt to the chosen blackbox model and get a
// non-streaming response back.

const DOWNSTREAM = process.env.DOWNSTREAM_URL ?? "http://localhost:8100";

export async function POST(req: NextRequest) {
  const body = await req.json();
  try {
    const r = await fetch(`${DOWNSTREAM}/generate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await r.text();
    return new NextResponse(text, {
      status: r.status,
      headers: { "content-type": r.headers.get("content-type") ?? "application/json" },
    });
  } catch (e: any) {
    return NextResponse.json({ error: `downstream unreachable: ${e.message}` }, { status: 502 });
  }
}
