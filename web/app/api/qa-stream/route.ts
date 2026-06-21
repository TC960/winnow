import { NextRequest } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { MODELS, type ModelId } from "@/lib/tokens";

// Streaming Q&A — used when only the compressed answer is wanted, rendered
// token-by-token. Useful as a faster single-answer mode if the A/B comparison
// is too noisy on stage.

const SYSTEM = `You are a careful assistant answering questions about a meeting transcript.
Answer ONLY using information present in the transcript provided. If the answer is
not in the transcript, reply exactly: "Not in transcript." Be concise.`;

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const { question, transcript, model } = (await req.json()) as {
    question: string;
    transcript: string;
    model?: ModelId;
  };
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return new Response("ANTHROPIC_API_KEY not set", { status: 500 });
  const modelId = MODELS[model ?? "sonnet-4-6"].anthropicId ?? "claude-sonnet-4-6";

  const client = new Anthropic({ apiKey });

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        const s = client.messages.stream({
          model: modelId,
          max_tokens: 256,
          system: SYSTEM,
          messages: [
            { role: "user", content: `Transcript:\n"""\n${transcript}\n"""\n\nQuestion: ${question}` },
          ],
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
