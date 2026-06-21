import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { MODELS, type ModelId } from "@/lib/tokens";

// A/B Q&A endpoint. Fires TWO Anthropic calls in parallel — one with the raw
// transcript, one with the compressed — using identical questions and the same
// strict "answer only from this transcript" instruction. The matching answers
// are the proof-of-fidelity: compression didn't drop the buried detail.

const SYSTEM = `You are a careful assistant answering questions about a meeting transcript.
Rules:
- Answer ONLY using information present in the transcript provided.
- If the answer is not in the transcript, reply exactly: "Not in transcript."
- Be concise (one or two sentences).
- Quote the specific detail when possible.`;

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const { question, raw, compressed, model } = (await req.json()) as {
    question: string;
    raw: string;
    compressed: string;
    model?: ModelId;
  };

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return NextResponse.json({ error: "ANTHROPIC_API_KEY not set" }, { status: 500 });

  const modelId = MODELS[model ?? "sonnet-4-6"].anthropicId ?? "claude-sonnet-4-6";
  const client = new Anthropic({ apiKey });

  const ask = async (transcript: string) => {
    const res = await client.messages.create({
      model: modelId,
      max_tokens: 256,
      system: SYSTEM,
      messages: [
        { role: "user", content: `Transcript:\n"""\n${transcript}\n"""\n\nQuestion: ${question}` },
      ],
    });
    const block = res.content[0];
    const text = block && block.type === "text" ? block.text : "";
    return {
      answer: text.trim(),
      usage: res.usage,
    };
  };

  try {
    const [rawRes, compRes] = await Promise.all([ask(raw), ask(compressed)]);
    return NextResponse.json({ raw: rawRes, compressed: compRes });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
