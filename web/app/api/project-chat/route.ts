import { NextRequest } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { MODELS, type ModelId } from "@/lib/tokens";

// Streaming chat over a "project" — a bundle of sources (the compressed
// transcript + any user-added text). Each turn:
//   1. Run the user's latest question through the FastAPI /compress_rag worker
//      (BGE reranker coarse selection + LLMLingua-2 token compression).
//   2. Feed the assembled, question-aware compressed prompt to Anthropic.
//   3. Stream the answer back over SSE, prefixed with a `meta` frame that
//      carries the compression stats so the UI can show savings per turn.

export const runtime = "nodejs";

const SYSTEM = `You are a study & sense-making assistant inside the user's project workspace.
You will receive a <sources> block that has already been compressed by an extractive
token classifier — the surface text may look terse or telegraphic, but the meaning is
intact. Treat every fact in it as authoritative.
- Ground every claim in the sources. Quote sparingly when it adds value.
- If something is NOT in the sources, say so plainly.
- Be concise. Use short paragraphs and tight bullet lists.`;

type Source = { title: string; content: string };

const BACKEND = process.env.COMPRESS_BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const { messages, sources, model, rate } = (await req.json()) as {
    messages: { role: "user" | "assistant"; content: string }[];
    sources: Source[];
    model?: ModelId;
    rate?: number;
  };

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return new Response("ANTHROPIC_API_KEY not set", { status: 500 });
  const modelId = MODELS[model ?? "sonnet-4-6"].anthropicId ?? "claude-sonnet-4-6";
  const client = new Anthropic({ apiKey });

  // Latest user turn drives question-aware compression.
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const question = lastUser?.content ?? "";
  const documents = sources.map((s) => `[${s.title}]\n${s.content}`).filter((s) => s.trim().length > 0);

  // Call the two-stage backend. If it fails or there's nothing to compress,
  // fall back to dumping sources verbatim so the user still gets an answer.
  let compressedBlock = "";
  let stats: any = null;
  let compressError: string | null = null;
  if (documents.length && question) {
    try {
      const r = await fetch(`${BACKEND}/compress_rag`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          instruction: "Use only the context to answer.",
          question,
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
  if (!compressedBlock) {
    // Fallback: raw sources verbatim. Cheap insurance against backend hiccups.
    compressedBlock = documents.join("\n\n");
  }

  // Build the messages: prior history stays, but the LATEST user message gets
  // augmented with the question-aware compressed context. This way the
  // Anthropic prompt cache still catches earlier turns.
  const augmentedMessages = messages.slice();
  const lastIdx = augmentedMessages.map((m) => m.role).lastIndexOf("user");
  if (lastIdx >= 0) {
    augmentedMessages[lastIdx] = {
      role: "user",
      content: `<sources>\n${compressedBlock}\n</sources>\n\n${augmentedMessages[lastIdx].content}`,
    };
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      // Send compression metadata up front so the UI can show the savings
      // even before Claude has emitted a single token.
      controller.enqueue(
        encoder.encode(`data: ${JSON.stringify({ meta: { stats, compressError } })}\n\n`)
      );
      try {
        const s = client.messages.stream({
          model: modelId,
          max_tokens: 1024,
          system: SYSTEM,
          messages: augmentedMessages.map((m) => ({ role: m.role, content: m.content })),
        });
        s.on("text", (chunk: string) => {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ delta: chunk })}\n\n`));
        });
        await s.finalMessage();
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ done: true })}\n\n`));
      } catch (e: any) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: e.message })}\n\n`));
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}
