type JsonValue = Record<string, unknown>;

async function jsonRequest<T>(url: string, payload: JsonValue): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || "请求失败");
  return data as T;
}

export type Chunk = { index: number; text: string };
export type Hit = {
  text: string;
  source: string;
  chunk_id?: string | number;
  file_type?: string;
  score?: number;
  rerank_score?: number;
};
export type VectorSummary = { index: number; dimension: number; preview: number[] };

export function processDocument(text: string) {
  return jsonRequest<{ input: string; cleaned_text: string; length: number }>(
    "/api/admin/offline/document-process",
    { text },
  );
}

export async function extractDocumentFile(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/admin/offline/document-file", {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(120_000),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || "文件解析失败");
  return data as { filename: string; file_type: string; content: string };
}

export function splitText(text: string, chunkSize: number, chunkOverlap: number) {
  return jsonRequest<{ count: number; chunks: Chunk[] }>("/api/admin/offline/split", {
    text,
    chunk_size: chunkSize,
    chunk_overlap: chunkOverlap,
  });
}

export function vectorizeChunks(texts: string[]) {
  return jsonRequest<{ vectors: VectorSummary[] }>("/api/admin/offline/vectorize", { texts });
}

export function storeChunks(chunks: string[], source: string, rebuild: boolean) {
  return jsonRequest<{ ok: boolean; chunks: number }>("/api/admin/offline/store", {
    chunks,
    source,
    rebuild,
  });
}

export function processQuery(query: string) {
  return jsonRequest<{ input: string; processed_query: string }>("/api/admin/online/query-process", {
    query,
  });
}

export function retrieveHits(query: string, topK: number, scoreThreshold: number) {
  return jsonRequest<{ query: string; hits: Hit[] }>("/api/admin/online/retrieve", {
    query,
    top_k: topK,
    score_threshold: scoreThreshold,
  });
}

export function rerankHits(query: string, hits: Hit[]) {
  return jsonRequest<{ query: string; hits: Hit[] }>("/api/admin/online/rerank", {
    query,
    hits,
  });
}

export function buildContext(hits: Hit[]) {
  return jsonRequest<{ context: string }>("/api/admin/online/context", { hits });
}

export function generateAnswer(message: string, threadId: string) {
  return jsonRequest<{ thread_id: string; answer: string }>("/api/admin/online/generate", {
    message,
    thread_id: threadId,
  });
}
