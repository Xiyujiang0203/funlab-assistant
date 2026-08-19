"use client";

import React, { useEffect, useState } from "react";
import {
  deleteDocument,
  getDocument,
  getRagConfig,
  listDocuments,
  previewSplit,
  rebuildRag,
  retrievePreview,
  saveRagConfig,
  updateDocument,
  uploadDocument,
  type RagConfig,
} from "@/lib/rag";

type Props = {
  open: boolean;
  onClose: () => void;
};

const defaultConfig: RagConfig = {
  rag_enabled: true,
  embed_model: "Pro/BAAI/bge-m3",
  rerank_model: "BAAI/bge-reranker-v2-m3",
  chunk_size: 300,
  chunk_overlap: 60,
  retrieve_top_k: 8,
  rerank_top_n: 5,
  knowledge_dir: "",
  embed_model_options: ["Pro/BAAI/bge-m3"],
  rerank_model_options: ["BAAI/bge-reranker-v2-m3", "Pro/BAAI/bge-reranker-v2-m3"],
};

export default function RagPanel({ open, onClose }: Props) {
  const [config, setConfig] = useState<RagConfig>(defaultConfig);
  const [documents, setDocuments] = useState<{ name: string; path: string; size: number }[]>([]);
  const [previewText, setPreviewText] = useState("");
  const [chunks, setChunks] = useState<{ index: number; text: string }[]>([]);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<{ text: string; source: string; score?: number; rerank_score?: number }[]>([]);
  const [selectedDoc, setSelectedDoc] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function refresh() {
    const [cfg, docs] = await Promise.all([getRagConfig(), listDocuments()]);
    setConfig(cfg);
    setDocuments(docs.documents);
  }

  useEffect(() => {
    if (!open) return;
    void refresh();
  }, [open]);

  async function handleSave() {
    setBusy(true);
    setMessage("");
    try {
      const saved = await saveRagConfig({
        rag_enabled: config.rag_enabled,
        embed_model: config.embed_model,
        rerank_model: config.rerank_model,
        chunk_size: Number(config.chunk_size),
        chunk_overlap: Number(config.chunk_overlap),
        retrieve_top_k: Number(config.retrieve_top_k),
        rerank_top_n: Number(config.rerank_top_n),
      });
      setConfig((prev) => ({ ...prev, ...saved }));
      setMessage("配置已保存");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "保存失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload(file: File | null) {
    if (!file) return;
    setBusy(true);
    setMessage("");
    try {
      const result = await uploadDocument(file);
      await refresh();
      setSelectedDoc(result.path);
      setMessage(`已上传 ${result.file}，新增 ${result.chunks} 个 chunk`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "上传失败");
    } finally {
      setBusy(false);
    }
  }

  async function handlePreviewSplit() {
    if (!previewText.trim()) {
      setMessage("请先输入要切分的文本");
      return;
    }
    setBusy(true);
    setMessage("");
    try {
      const result = await previewSplit(previewText, Number(config.chunk_size), Number(config.chunk_overlap));
      setChunks(result.chunks);
      setMessage(`切分完成，共 ${result.count} 个 chunk`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "预览失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleLoadDocument() {
    if (!selectedDoc) {
      setMessage("请先选择文档");
      return;
    }
    setBusy(true);
    setMessage("");
    try {
      const result = await getDocument(selectedDoc);
      setPreviewText(result.content);
      setMessage(`已载入 ${result.name}`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "载入失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleSaveDocument() {
    if (!selectedDoc) {
      setMessage("请先选择文档");
      return;
    }
    setBusy(true);
    setMessage("");
    try {
      await updateDocument(selectedDoc, previewText);
      await refresh();
      setMessage("文档已保存");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "保存文档失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteDocument() {
    if (!selectedDoc) {
      setMessage("请先选择文档");
      return;
    }
    setBusy(true);
    setMessage("");
    try {
      await deleteDocument(selectedDoc);
      await refresh();
      setSelectedDoc("");
      setPreviewText("");
      setChunks([]);
      setMessage("文档已删除");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "删除失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleRetrievePreview() {
    if (!query.trim()) {
      setMessage("请先输入检索问题");
      return;
    }
    setBusy(true);
    setMessage("");
    try {
      const result = await retrievePreview(query, Number(config.retrieve_top_k));
      setHits(result.hits);
      setMessage(`检索完成，共 ${result.hits.length} 条`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "检索失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleRebuild() {
    setBusy(true);
    setMessage("");
    try {
      const result = await rebuildRag();
      setMessage(`索引已重建，共 ${result.chunks} 个 chunk`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "重建失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <aside className={`rag-panel${open ? " open" : ""}`}>
      <div className="rag-panel-header">
        <h2 className="rag-panel-title">RAG 面板</h2>
        <button type="button" className="rag-close-btn" onClick={onClose}>关闭</button>
      </div>

      <div className="rag-section">
        <label className="rag-row-inline">
          <span>启用 RAG</span>
          <input
            type="checkbox"
            checked={config.rag_enabled}
            onChange={(e) => setConfig((prev) => ({ ...prev, rag_enabled: e.target.checked }))}
          />
        </label>
        <label className="rag-field">
          <span>Embedding 模型</span>
          <select value={config.embed_model} onChange={(e) => setConfig((prev) => ({ ...prev, embed_model: e.target.value }))}>
            {config.embed_model_options.map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
        </label>
        <label className="rag-field">
          <span>重排模型</span>
          <select value={config.rerank_model} onChange={(e) => setConfig((prev) => ({ ...prev, rerank_model: e.target.value }))}>
            {config.rerank_model_options.map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
        </label>
        <div className="rag-grid">
          <label className="rag-field">
            <span>chunk size</span>
            <input type="number" value={config.chunk_size} onChange={(e) => setConfig((prev) => ({ ...prev, chunk_size: Number(e.target.value) }))} />
          </label>
          <label className="rag-field">
            <span>chunk overlap</span>
            <input type="number" value={config.chunk_overlap} onChange={(e) => setConfig((prev) => ({ ...prev, chunk_overlap: Number(e.target.value) }))} />
          </label>
          <label className="rag-field">
            <span>retrieve topK</span>
            <input type="number" value={config.retrieve_top_k} onChange={(e) => setConfig((prev) => ({ ...prev, retrieve_top_k: Number(e.target.value) }))} />
          </label>
          <label className="rag-field">
            <span>rerank topN</span>
            <input type="number" value={config.rerank_top_n} onChange={(e) => setConfig((prev) => ({ ...prev, rerank_top_n: Number(e.target.value) }))} />
          </label>
        </div>
        <button type="button" className="rag-primary-btn" onClick={() => void handleSave()} disabled={busy}>
          保存配置
        </button>
      </div>

      <div className="rag-section">
        <div className="rag-section-title">加入文档</div>
        <input type="file" onChange={(e) => void handleUpload(e.target.files?.[0] || null)} />
        <button type="button" className="rag-secondary-btn" onClick={() => void handleRebuild()} disabled={busy}>
          重建索引
        </button>
        <ul className="rag-doc-list">
          {documents.map((doc) => (
            <li key={doc.path} className="rag-doc-item">
              <span>{doc.name}</span>
              <small>{Math.ceil(doc.size / 1024)} KB</small>
            </li>
          ))}
        </ul>
      </div>

      <div className="rag-section">
        <div className="rag-section-title">预览切分 chunk</div>
        <label className="rag-field">
          <span>选择文档</span>
          <select value={selectedDoc} onChange={(e) => setSelectedDoc(e.target.value)}>
            <option value="">请选择文档</option>
            {documents.map((doc) => (
              <option key={doc.path} value={doc.path}>{doc.name}</option>
            ))}
          </select>
        </label>
        <div className="rag-action-row">
          <button type="button" className="rag-secondary-btn" onClick={() => void handleLoadDocument()} disabled={busy}>
            载入文档
          </button>
          <button type="button" className="rag-secondary-btn" onClick={() => void handleSaveDocument()} disabled={busy}>
            保存文档
          </button>
          <button type="button" className="rag-danger-btn" onClick={() => void handleDeleteDocument()} disabled={busy}>
            删除文档
          </button>
        </div>
        <textarea
          className="rag-textarea"
          value={previewText}
          onChange={(e) => setPreviewText(e.target.value)}
          placeholder="粘贴文本或先载入文档，再预览切分结果"
        />
        <button type="button" className="rag-secondary-btn" onClick={() => void handlePreviewSplit()} disabled={busy}>
          预览切分
        </button>
        <div className="rag-chunk-list">
          {chunks.map((chunk) => (
            <div key={chunk.index} className="rag-chunk-item">
              <strong>Chunk {chunk.index + 1}</strong>
              <p>{chunk.text}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="rag-section">
        <div className="rag-section-title">预览检索 + 重排</div>
        <input
          className="rag-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="输入问题查看召回结果"
        />
        <button type="button" className="rag-secondary-btn" onClick={() => void handleRetrievePreview()} disabled={busy}>
          预览检索
        </button>
        <div className="rag-hit-list">
          {hits.map((hit, idx) => (
            <div key={`${hit.source}-${idx}`} className="rag-hit-item">
              <div className="rag-hit-meta">
                <span>{hit.source.split("/").pop()}</span>
                <small>vec:{hit.score?.toFixed?.(3) ?? "-"} / rerank:{hit.rerank_score?.toFixed?.(3) ?? "-"}</small>
              </div>
              <p>{hit.text}</p>
            </div>
          ))}
        </div>
      </div>

      {message ? <div className="rag-message">{message}</div> : null}
    </aside>
  );
}

