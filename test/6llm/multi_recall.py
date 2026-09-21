"""② 多路召回：原问题 / 改写 / 关键词 / 子问题 → 向量检索 → RRF 合并。"""
from __future__ import annotations

import requests
from qdrant_client import QdrantClient

from query_rewrite import QueryPlan

EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBED_MODEL = "text-embedding-v4"


def embed_query(query: str, api_key: str, dimensions: int = 1024) -> list[float]:
    resp = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": EMBED_MODEL,
            "input": query,
            "encoding_format": "float",
            "dimensions": dimensions,
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
    return [{**best[key], "rrf_score": rrf} for key, rrf in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def _recall_one(
    query: str,
    *,
    qdrant: QdrantClient,
    collection: str,
    api_key: str,
    top_k: int,
    path: str,
) -> list[dict]:
    if not query.strip():
        return []
    vec = embed_query(query, api_key)
    rows = search_qdrant(qdrant, collection, vec, top_k)
    for r in rows:
        r["recall_path"] = path
    return rows


def build_path_queries(plan: QueryPlan) -> list[tuple[str, str]]:
    """返回 [(path_name, query_text), ...]。"""
    paths: list[tuple[str, str]] = [("original", plan.original)]
    if plan.rewrite and plan.rewrite != plan.original:
        paths.append(("rewrite", plan.rewrite))
    kw = plan.keyword_query
    if kw:
        paths.append(("keywords", kw))
    for i, sub in enumerate(plan.sub_queries, 1):
        paths.append((f"sub_{i}", sub))
    return paths


def multi_recall(
    plan: QueryPlan,
    *,
    qdrant: QdrantClient,
    collection: str,
    api_key: str,
    recall_top: int = 8,
    rrf_k: int = 60,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """
    四类检索：原问题、改写问题、关键词、子问题；RRF 合并。
    返回 (fused_docs, path_queries)。
    """
    path_queries = build_path_queries(plan)
    lists: list[list[dict]] = []
    for path, q in path_queries:
        lists.append(
            _recall_one(
                q,
                qdrant=qdrant,
                collection=collection,
                api_key=api_key,
                top_k=recall_top,
                path=path,
            )
        )
    return rrf_fuse(lists, k=rrf_k), path_queries
