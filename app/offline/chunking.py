"""Offline chunking (from test/2chunk)."""
from __future__ import annotations

import re

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def split_structure(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)
    docs = splitter.split_text(text)
    chunks: list[dict] = []
    overflow = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
    )
    for doc in docs:
        meta = dict(doc.metadata)
        content = doc.page_content.strip()
        if not content:
            continue
        if len(content) <= chunk_size:
            chunks.append({"text": content, "meta": meta})
            continue
        for part in overflow.split_text(content):
            if part.strip():
                chunks.append({"text": part.strip(), "meta": meta})
    return chunks


def split_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "！", "？", " ", ""],
    )
    return [{"text": c, "meta": {}} for c in splitter.split_text(text) if c.strip()]


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    strategy: str = "auto",
) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    if strategy == "auto":
        strategy = "structure" if re.search(r"(?m)^#{1,4}\s+", text) else "prose"
    if strategy == "structure":
        return split_structure(text, chunk_size, chunk_overlap)
    return split_recursive(text, chunk_size, chunk_overlap)


def split_documents(
    documents: list[dict],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict]:
    out: list[dict] = []
    for doc in documents:
        parts = chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, part in enumerate(parts, 1):
            meta = part.get("meta") or {}
            if isinstance(meta, dict):
                meta_s = ", ".join(f"{k}={v}" for k, v in meta.items())
            else:
                meta_s = str(meta)
            out.append(
                {
                    "source": doc.get("source", "unknown"),
                    "chunk_id": i,
                    "chars": len(part["text"]),
                    "meta": meta_s,
                    "text": part["text"],
                }
            )
    return out
