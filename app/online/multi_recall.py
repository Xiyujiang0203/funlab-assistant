"""② Multi-path recall + RRF (from test/6llm/multi_recall.py)."""
from __future__ import annotations

import requests
from qdrant_client import QdrantClient

from app.config import get_settings
from app.online.query_rewrite import QueryPlan
from app.qdrant_client import get_qdrant

EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"


def embed_query(query: str, dimensions: int | None = None) -> list[float]:
    s = get_settings()
    if not s.dashscope_api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY")
    resp = requests.post(
        EMBED_URL,
        headers={
            "Authorization": f"Bearer {s.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": s.embed_model,
            "input": query,
            "encoding_format": "float",
            "dimensions": dimensions or s.embed_dim,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"embed failed {resp.status_code}: {resp.text[:300]}")
    return resp.json()["data"][0]["embedding"]


def search_qdrant(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    top_k: int,
) -> list[dict]:
    result = client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    rows = []
    for h in result.points:
        payload = h.payload or {}
        rows.append(
            {
                "id": str(h.id),
                "score": float(h.score or 0.0),
                "source": payload.get("source", ""),
                "chunk_id": payload.get("chunk_id"),
                "meta": payload.get("meta", ""),
                "text": payload.get("text", ""),
            }
        )
    return rows


def doc_key(d: dict) -> str:
    return f"{d.get('source', '')}::{d.get('chunk_id')}"


def rrf_fuse(rank_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    best: dict[str, dict] = {}
    for docs in rank_lists:
        for rank, d in enumerate(docs, 1):
            key = doc_key(d)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in best or float(d.get("score") or 0) > float(best[key].get("score") or 0):
                best[key] = d
    return [
        {**best[key], "rrf_score": rrf}
        for key, rrf in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


def build_path_queries(plan: QueryPlan) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = [("original", plan.original)]
    if plan.rewrite and plan.rewrite != plan.original:
        paths.append(("rewrite", plan.rewrite))
    kw = plan.keyword_query
    if kw:
        paths.append(("keywords", kw))
    for i, sub in enumerate(plan.sub_queries, 1):
        paths.append((f"sub_{i}", sub))
    return paths


def _recall_one(query: str, *, collection: str, top_k: int, path: str) -> list[dict]:
    if not query.strip():
        return []
    vec = embed_query(query)
    rows = search_qdrant(get_qdrant(), collection, vec, top_k)
    for r in rows:
        r["recall_path"] = path
    return rows


def multi_recall(
    plan: QueryPlan,
    *,
    recall_top: int = 8,
    rrf_k: int = 60,
) -> tuple[list[dict], list[tuple[str, str]]]:
    collection = get_settings().qdrant_collection
    path_queries = build_path_queries(plan)
    lists = [
        _recall_one(q, collection=collection, top_k=recall_top, path=path)
        for path, q in path_queries
    ]
    return rrf_fuse(lists, k=rrf_k), path_queries


def retrieve_hits(query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:
    """Simple single-path recall for admin preview."""
    from app.online.query_rewrite import QueryPlan as QP

    fused, _ = multi_recall(QP(original=query, rewrite=query), recall_top=top_k)
    out = []
    for d in fused[:top_k]:
        if float(d.get("score") or 0) < score_threshold and "rrf_score" not in d:
            continue
        out.append(d)
    return out
