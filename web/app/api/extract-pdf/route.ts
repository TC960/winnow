import { NextRequest, NextResponse } from "next/server";

// Server-side PDF text extraction. The client POSTs the PDF as multipart
// form-data (field: "file"); we run pdf-parse against the bytes and return
// the extracted text + page count + char count. Keeping it server-side
// avoids shipping a pdf.js bundle to the browser.

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
  const buf = Buffer.from(await file.arrayBuffer());

  try {
    const { PDFParse } = await import("pdf-parse");
    const parser = new PDFParse({ data: new Uint8Array(buf) });
    const out = await parser.getText();
    await parser.destroy();
    const text = String(out.text ?? "").trim();
    const pages = (out as any).total ?? (out as any).pages?.length;
    return NextResponse.json({ text, pages, chars: text.length, filename: file.name });
  } catch (e: any) {
    return NextResponse.json({ error: `pdf parse failed: ${e.message}` }, { status: 500 });
  }
}
