import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { MODELS, type ModelId } from "@/lib/tokens";

// Structured one-shot actions over the same project sources. Each action has
// its own tight prompt and expected output shape. Returned JSON gets cached
// in the store so flipping between insights is instant.

export const runtime = "nodejs";

type Source = { title: string; content: string };
export type Action = "summary" | "decisions" | "actions" | "flashcards" | "glossary";

const ACTION_PROMPTS: Record<Action, { instruction: string; jsonSchema: string }> = {
  summary: {
    instruction: "Write a tight 3-5 sentence summary of the sources. No bullet lists, just prose.",
    jsonSchema: `{ "summary": string }`,
  },
  decisions: {
    instruction: "List the concrete DECISIONS that were made (not topics discussed). Each item must be a specific resolution.",
    jsonSchema: `{ "decisions": string[] }`,
  },
  actions: {
    instruction: "List the action items. Each item should include WHO is responsible (if mentioned) and a clear next step.",
    jsonSchema: `{ "actions": string[] }`,
  },
  flashcards: {
    instruction: "Generate 5-8 study flashcards covering the key facts a reader should remember. Question + concise answer.",
    jsonSchema: `{ "flashcards": [{ "q": string, "a": string }] }`,
  },
  glossary: {
    instruction: "Extract domain terms or names mentioned and define each in one sentence using only the sources.",
    jsonSchema: `{ "glossary": [{ "term": string, "definition": string }] }`,
  },
};

export async function POST(req: NextRequest) {
  const { action, sources, model } = (await req.json()) as {
    action: Action;
    sources: Source[];
    model?: ModelId;
  };

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return NextResponse.json({ error: "ANTHROPIC_API_KEY not set" }, { status: 500 });
  const modelId = MODELS[model ?? "sonnet-4-6"].anthropicId ?? "claude-sonnet-4-6";
  const spec = ACTION_PROMPTS[action];
  if (!spec) return NextResponse.json({ error: `unknown action: ${action}` }, { status: 400 });

  const client = new Anthropic({ apiKey });
  const sourcesBlock = sources.map((s, i) =>
    `<source index="${i + 1}" title="${escapeXml(s.title)}">\n${s.content}\n</source>`
  ).join("\n\n");

  try {
    const res = await client.messages.create({
      model: modelId,
      max_tokens: 1024,
      system: [
        { type: "text", text: "You output ONLY valid JSON matching the schema given. No prose, no code fences." },
        { type: "text", text: `SOURCES:\n\n${sourcesBlock}`, cache_control: { type: "ephemeral" } },
      ],
      messages: [
        {
          role: "user",
          content: `${spec.instruction}\n\nReturn JSON exactly matching: ${spec.jsonSchema}`,
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
    return NextResponse.json({ result: parsed });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}

function escapeXml(s: string) {
  return s.replace(/[<>&"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", "\"": "&quot;" }[c]!));
}
