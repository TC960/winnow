import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { MODELS, type ModelId } from "@/lib/tokens";

// Structured one-shot actions over the project sources. Each action runs the
// same two-stage compressor (BGE reranker + LLMLingua-2) against the sources,
// using the action's own instruction as the "query" so the reranker biases
// toward the parts of the transcript most relevant to e.g. "decisions" or
// "action items". The compressed bundle is then handed to Claude with a
// schema-locked prompt.

export const runtime = "nodejs";

type Source = { title: string; content: string };
export type Action = "summary" | "decisions" | "actions" | "flashcards" | "glossary";

const BACKEND = process.env.COMPRESS_BACKEND_URL ?? "http://localhost:8000";

const ACTION_PROMPTS: Record<Action, { question: string; instruction: string; jsonSchema: string }> = {
  summary: {
    question: "What is the overall point and what were the main topics covered?",
    instruction: "Write a tight 3-5 sentence summary of the sources. No bullet lists, just prose.",
    jsonSchema: `{ "summary": string }`,
  },
  decisions: {
    question: "What concrete decisions, resolutions, or sign-offs were made?",
    instruction: "List the concrete DECISIONS that were made (not topics discussed). Each item must be a specific resolution.",
    jsonSchema: `{ "decisions": string[] }`,
  },
  actions: {
    question: "What action items were assigned, by when, and to whom?",
    instruction: "List the action items. Each item should include WHO is responsible (if mentioned) and a clear next step.",
    jsonSchema: `{ "actions": string[] }`,
  },
  flashcards: {
    question: "What are the most testable, memorable facts from these sources?",
    instruction: "Generate 5-8 study flashcards covering the key facts a reader should remember. Question + concise answer.",
    jsonSchema: `{ "flashcards": [{ "q": string, "a": string }] }`,
  },
  glossary: {
    question: "What domain terms, jargon, names, and acronyms appear in the sources?",
    instruction: "Extract domain terms or names mentioned and define each in one sentence using only the sources.",
    jsonSchema: `{ "glossary": [{ "term": string, "definition": string }] }`,
  },
};

export async function POST(req: NextRequest) {
  const { action, sources, model, rate } = (await req.json()) as {
    action: Action;
    sources: Source[];
    model?: ModelId;
    rate?: number;
  };

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return NextResponse.json({ error: "ANTHROPIC_API_KEY not set" }, { status: 500 });
  const modelId = MODELS[model ?? "sonnet-4-6"].anthropicId ?? "claude-sonnet-4-6";
  const spec = ACTION_PROMPTS[action];
  if (!spec) return NextResponse.json({ error: `unknown action: ${action}` }, { status: 400 });

  const documents = sources
    .map((s) => `[${s.title}]\n${s.content}`)
    .filter((s) => s.trim().length > 0);

  // Compress through the backend. Each insight uses its own "question" so the
  // reranker selects the most relevant chunks for that specific task.
  let compressedBlock = "";
  let stats: any = null;
  let compressError: string | null = null;
  if (documents.length) {
    try {
      const r = await fetch(`${BACKEND}/compress_rag`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          instruction: spec.instruction,
          question: spec.question,
          documents,
          rate: rate ?? 0.5,
        }),
      });
      if (r.ok) {
        const j = await r.json();
        compressedBlock = j.compressed_context ?? "";
        stats = {
          contextOriginTokens: j.context_origin_tokens,
          contextCompressedTokens: j.context_compressed_tokens,
          originTokens: j.origin_tokens,
          compressedTokens: j.compressed_tokens,
          rate: j.rate,
          keptDocuments: j.kept_documents,
          totalDocuments: j.total_documents,
          rerankerScores: j.reranker_scores,
        };
      } else {
        compressError = `compress_rag ${r.status}`;
      }
    } catch (e: any) {
      compressError = e.message;
    }
  }
  if (!compressedBlock) compressedBlock = documents.join("\n\n");

  const client = new Anthropic({ apiKey });

  try {
    const res = await client.messages.create({
      model: modelId,
      max_tokens: 1024,
      system: "You output ONLY valid JSON matching the schema given. No prose, no code fences.",
      messages: [
        {
          role: "user",
          content:
            `<sources>\n${compressedBlock}\n</sources>\n\n` +
            `${spec.instruction}\n\nReturn JSON exactly matching: ${spec.jsonSchema}`,
        },
      ],
    });
    const block = res.content[0];
    const text = (block && block.type === "text" ? block.text : "").trim();
    const cleaned = text.replace(/^```json\s*|\s*```$/g, "");
    let parsed: any;
    try {
      parsed = JSON.parse(cleaned);
    } catch {
      return NextResponse.json({ error: "model returned non-JSON", raw: text }, { status: 502 });
    }
    return NextResponse.json({ result: parsed, stats, compressError });
  } catch (e: any) {
    return NextResponse.json({ error: e.message, stats, compressError }, { status: 500 });
  }
}
