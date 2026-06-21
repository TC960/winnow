import { NextRequest } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { MODELS, type ModelId } from "@/lib/tokens";

// Streaming chat over a "project" — a bundle of sources (the compressed
// transcript + any user-added text). Prompt-cacheable: the big static block
// of sources is sent with `cache_control: ephemeral` so repeated questions in
// the same session hit the Anthropic prompt cache and cost ~1/10th.

export const runtime = "nodejs";

const SYSTEM = `You are a study & sense-making assistant inside the user's project workspace.
You have access to a transcript and (optionally) other source documents.
- Ground every claim in the sources. Quote sparingly when it adds value.
- If something is NOT in the sources, say so plainly.
- Be concise. Use short paragraphs and tight bullet lists.`;

type Source = { title: string; content: string };

export async function POST(req: NextRequest) {
  const { messages, sources, model } = (await req.json()) as {
    messages: { role: "user" | "assistant"; content: string }[];
    sources: Source[];
    model?: ModelId;
  };

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return new Response("ANTHROPIC_API_KEY not set", { status: 500 });
  const modelId = MODELS[model ?? "sonnet-4-6"].anthropicId ?? "claude-sonnet-4-6";

  const client = new Anthropic({ apiKey });

  // Pack sources as one static block, cached.
  const sourcesBlock = sources.map((s, i) =>
    `<source index="${i + 1}" title="${escapeXml(s.title)}">\n${s.content}\n</source>`
  ).join("\n\n");

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        const s = client.messages.stream({
          model: modelId,
          max_tokens: 1024,
          system: [
            { type: "text", text: SYSTEM },
            { type: "text", text: `SOURCES:\n\n${sourcesBlock}`, cache_control: { type: "ephemeral" } },
          ],
          messages: messages.map((m) => ({ role: m.role, content: m.content })),
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

function escapeXml(s: string) {
  return s.replace(/[<>&"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", "\"": "&quot;" }[c]!));
}
