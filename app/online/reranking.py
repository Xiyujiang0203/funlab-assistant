"""Qwen rerank (from test/5reranker)."""
from __future__ import annotations

import requests

from app.config import get_settings

RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


def rerank_hits(query: str, docs: list[dict], top_n: int | None = None) -> list[dict]:
    if not docs:
        return []
    s = get_settings()
    if not s.dashscope_api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY")
    n = top_n or len(docs)
    documents = [d.get("text", "") for d in docs]
    resp = requests.post(
        RERANK_URL,
        headers={
            "Authorization": f"Bearer {s.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": s.rerank_model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": min(n, len(documents)), "return_documents": False},
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"rerank failed {resp.status_code}: {resp.text[:300]}")
    results = resp.json().get("output", {}).get("results", [])
    ranked = []
    for item in results:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(docs):
            ranked.append({**docs[idx], "rerank_score": item.get("relevance_score")})
    return ranked
