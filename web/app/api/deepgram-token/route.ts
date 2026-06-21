import { NextResponse } from "next/server";

// Return a Deepgram token the browser can use for WebSocket subprotocol auth.
// Preferred path: mint a short-lived token via /v1/auth/grant. That requires
// the project API key to have admin scope. Fallback: return the raw API key —
// fine for a hackathon demo where everything is local; do NOT ship to prod.

export async function POST() {
  const key = process.env.DEEPGRAM_API_KEY;
  if (!key) return NextResponse.json({ error: "DEEPGRAM_API_KEY not set" }, { status: 500 });

  try {
    const r = await fetch("https://api.deepgram.com/v1/auth/grant", {
      method: "POST",
      headers: { Authorization: `Token ${key}`, "content-type": "application/json" },
    });
    if (r.ok) {
      const data = await r.json();
      return NextResponse.json({ access_token: data.access_token, expires_in: data.expires_in });
    }
    // 403 / 401: this API key isn't allowed to grant tokens. Fall back to the raw key.
    return NextResponse.json({ access_token: key, expires_in: 30 });
  } catch {
    return NextResponse.json({ access_token: key, expires_in: 30 });
  }
}
