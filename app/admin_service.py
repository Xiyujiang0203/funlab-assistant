"""Admin preview helpers wired to test pipeline modules."""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.offline.chunking import chunk_text
from app.offline.document_processing import load_document
from app.offline.vector_db import store_text_chunks as _store_text_chunks
from app.offline.vectorization import embed_texts
from app.online.context_expand import build_context, expand_context
from app.online.multi_recall import retrieve_hits
from app.online.pipeline import run_online_retrieval, run_rag_answer
from app.online.query_rewrite import plan_query
from app.online.reranking import rerank_hits
from app.rag_state import get_rag_runtime

SUPPORTED_ADMIN_UPLOAD_EXTENSIONS = {".md", ".pdf", ".txt"}


def process_document_text(text: str) -> dict:
    cleaned = (text or "").strip()
    return {"input": text, "cleaned_text": cleaned, "length": len(cleaned)}


def extract_uploaded_document(filename: str, content: bytes) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_ADMIN_UPLOAD_EXTENSIONS:
        raise ValueError("仅支持 PDF / Markdown / TXT")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp.flush()
        temp_path = Path(tmp.name)
    try:
        doc = load_document(temp_path)
        if not doc:
            raise ValueError("文件无有效文本")
        return {
            "filename": Path(filename).name,
            "file_type": suffix,
            "content": doc["text"],
        }
    finally:
        temp_path.unlink(missing_ok=True)


def split_text_preview(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> dict:
    runtime = get_rag_runtime()
    parts = chunk_text(
        text,
        chunk_size=chunk_size or runtime.chunk_size,
        chunk_overlap=chunk_overlap or runtime.chunk_overlap,
    )
    return {
        "count": len(parts),
        "chunks": [{"index": i, "text": p["text"], "meta": p.get("meta")} for i, p in enumerate(parts, 1)],
    }


def vectorize_texts_preview(texts: list[str]) -> dict:
    vectors = embed_texts(texts)
    return {
        "vectors": [
            {"index": i, "dim": len(v), "preview": v[:8]}
            for i, v in enumerate(vectors, 1)
        ]
    }


def store_text_chunks(chunks: list[str], source: str = "manual", rebuild: bool = False) -> dict:
    n = _store_text_chunks(chunks, source=source, rebuild=rebuild)
    return {"ok": True, "chunks": n}


def process_query_preview(query: str) -> dict:
    plan = plan_query(query)
    return {
        "input": query,
        "processed_query": plan.rewrite,
        "keywords": plan.keywords,
        "sub_queries": plan.sub_queries,
    }


def retrieve_preview(query: str, top_k: int = 5, score_threshold: float = 0.0) -> dict:
    hits = retrieve_hits(query, top_k=top_k, score_threshold=score_threshold)
    return {"query": query, "hits": hits}


def rerank_preview(query: str, hits: list[dict]) -> dict:
    ranked = rerank_hits(query, hits)
    return {"query": query, "hits": ranked}


def build_context_preview(hits: list[dict]) -> dict:
    runtime = get_rag_runtime()
    expanded = expand_context(hits, window=runtime.context_window)
    return {"context": build_context(expanded), "windows": expanded}


def generate_preview(query: str, thread_id: str = "") -> dict:
    answer = run_rag_answer(query, stream=False)
    return {"thread_id": thread_id, "answer": answer}
