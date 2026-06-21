// Trigger a browser download for an in-memory string. Used by the Test-tab
// cards to export the compressed prompt as a .md/.txt file.

export function downloadText(filename: string, content: string, mime = "text/markdown") {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function slugify(s: string): string {
  return (s || "winnow").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "winnow";
}
