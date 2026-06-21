import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.COMPRESS_BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.json();
  try {
    const r = await fetch(`${BACKEND}/trace/pack`, {
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
    return NextResponse.json({ error: `backend unreachable: ${e.message}` }, { status: 502 });
  }
}
