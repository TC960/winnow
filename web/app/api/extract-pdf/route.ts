import { NextRequest, NextResponse } from "next/server";

// Server-side PDF text extraction via unpdf (a Next.js-friendly pdfjs-dist
// wrapper). Client POSTs the PDF as multipart form-data (field: "file");
// we return the extracted text + page count.

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  let form: FormData;
  try {
    form = await req.formData();
  } catch (e: any) {
    return NextResponse.json({ error: `bad form-data: ${e.message}` }, { status: 400 });
  }
  const file = form.get("file");
  if (!(file instanceof File)) {
    return NextResponse.json({ error: "missing 'file' field" }, { status: 400 });
  }
  const bytes = new Uint8Array(await file.arrayBuffer());

  try {
    const { extractText, getDocumentProxy } = await import("unpdf");
    const pdf = await getDocumentProxy(bytes);
    const out = await extractText(pdf, { mergePages: true });
    const text = String(out.text ?? "").trim();
    return NextResponse.json({
      text,
      pages: out.totalPages,
      chars: text.length,
      filename: file.name,
    });
  } catch (e: any) {
    return NextResponse.json({ error: `pdf parse failed: ${e.message}` }, { status: 500 });
  }
}
