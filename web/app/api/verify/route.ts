import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { MODELS, type ModelId } from "@/lib/tokens";

// Trace-mode fidelity check. Asks Claude the SAME question twice in parallel, once
// on the full history and once on the compact (packed) history, then grades the two
// answers SEMANTICALLY: do they convey the same key fact? This mirrors the grader in
// trace/eval.py (Claude as a strict judge), replacing the brittle string-equality
// check the UI used before. Two correct-but-differently-worded answers now count as
// aligned; only a genuine change in the conveyed fact counts as diverged.

const ANSWER_SYSTEM = `You are a careful assistant answering questions about a conversation history.
Rules:
- Answer ONLY using information present in the context provided.
- If the answer is not in the context, reply exactly: "Not in context."
- Be concise (one or two sentences).
- Quote the specific detail when possible.`;

const GRADER_SYSTEM = `You are a strict grader comparing two answers to the same question.
Reply with exactly one word: ALIGNED if both answers convey the same key fact (or both
correctly state the information is absent), otherwise DIVERGED. Treat one answer having a
specific fact and the other saying "Not in context" as DIVERGED.`;

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

  const ask = async (system: string, user: string) => {
    const res = await client.messages.create({
      model: modelId,
      max_tokens: 256,
      temperature: 0,
      system,
      messages: [{ role: "user", content: user }],
    });
    const block = res.content[0];
    return block && block.type === "text" ? block.text.trim() : "";
  };

  const answer = (context: string) =>
    ask(ANSWER_SYSTEM, `Context:\n"""\n${context}\n"""\n\nQuestion: ${question}`);

  try {
    const [rawAnswer, compactAnswer] = await Promise.all([answer(raw), answer(compressed)]);
    // Semantic grade: one extra Claude call, temperature 0, identical to the eval's
    // judge so the badge means what people expect.
    const verdict = await ask(
      GRADER_SYSTEM,
      `Question: ${question}\nAnswer A (full history): ${rawAnswer}\n` +
        `Answer B (compact history): ${compactAnswer}\n\nReply ALIGNED or DIVERGED.`
    );
    const aligned = verdict.toUpperCase().startsWith("ALIGNED");
    return NextResponse.json({
      raw: { answer: rawAnswer },
      compact: { answer: compactAnswer },
      aligned,
    });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
