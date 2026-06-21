import { NextRequest, NextResponse } from "next/server";

// Thin proxy to the FastAPI/Modal compression worker. Keeps the Modal app and
// its API key off the browser, and avoids CORS by going through same-origin.

const BACKEND = process.env.COMPRESS_BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.json();
  try {
    const r = await fetch(`${BACKEND}/compress`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        text: body.text,
        rate: body.rate ?? 0.5,
        return_labels: body.return_labels ?? true,
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
