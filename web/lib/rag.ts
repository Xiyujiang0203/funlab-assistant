export type RagConfig = {
  rag_enabled: boolean;
  embed_model: string;
  rerank_model: string;
  chunk_size: number;
  chunk_overlap: number;
  retrieve_top_k: number;
  rerank_top_n: number;
  knowledge_dir: string;
  embed_model_options: string[];
  rerank_model_options: string[];
};

async function jsonRequest<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || "请求失败");
  return data as T;
}

export function getRagConfig() {
  return jsonRequest<RagConfig>("/api/rag/config");
}

export function saveRagConfig(payload: Omit<RagConfig, "knowledge_dir" | "embed_model_options" | "rerank_model_options">) {
  return jsonRequest<RagConfig>("/api/rag/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function previewSplit(text: string, chunk_size: number, chunk_overlap: number) {
  return jsonRequest<{ count: number; chunks: { index: number; text: string }[] }>("/api/rag/preview-split", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, chunk_size, chunk_overlap }),
  });
}

export function retrievePreview(query: string, top_k: number) {
  return jsonRequest<{ hits: { text: string; source: string; score?: number; rerank_score?: number }[] }>("/api/rag/retrieve-preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k }),
  });
}

export function listDocuments() {
  return jsonRequest<{ documents: { name: string; path: string; size: number }[] }>("/api/rag/documents");
}

export function getDocument(path: string) {
  return jsonRequest<{ name: string; path: string; content: string }>(`/api/rag/document?path=${encodeURIComponent(path)}`);
}

export function updateDocument(path: string, content: string) {
  return jsonRequest<{ ok: boolean; path: string }>(`/api/rag/document?path=${encodeURIComponent(path)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
}

export function deleteDocument(path: string) {
  return jsonRequest<{ ok: boolean; path: string }>(`/api/rag/document?path=${encodeURIComponent(path)}`, {
    method: "DELETE",
  });
}

export async function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/rag/documents", { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || "上传失败");
  return data as { ok: boolean; file: string; path: string; chunks: number };
}

export function rebuildRag() {
  return jsonRequest<{ ok: boolean; chunks: number }>("/api/rag/rebuild", { method: "POST" });
}

