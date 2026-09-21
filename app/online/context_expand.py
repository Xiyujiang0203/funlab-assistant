"""③ Context window expand (from test/6llm/context_expand.py)."""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings
from app.qdrant_client import get_qdrant


def _payload_row(point, *, role: str) -> dict:
    payload = point.payload or {}
    return {
        "id": str(point.id),
        "source": payload.get("source", ""),
        "chunk_id": int(payload.get("chunk_id")),
        "meta": payload.get("meta", ""),
        "text": payload.get("text", ""),
        "role": role,
    }


def _merge_intervals(intervals: list[tuple[str, int, int, dict]]) -> list[dict]:
    by_src: dict[str, list[tuple[int, int, dict]]] = {}
    for source, lo, hi, seed in intervals:
        by_src.setdefault(source, []).append((lo, hi, seed))
    windows: list[dict] = []
    for source, items in by_src.items():
        items.sort(key=lambda x: x[0])
        cur_lo, cur_hi, seeds = items[0][0], items[0][1], [items[0][2]]
        for lo, hi, seed in items[1:]:
            if lo <= cur_hi + 1:
                cur_hi = max(cur_hi, hi)
                seeds.append(seed)
            else:
                windows.append({"source": source, "lo": cur_lo, "hi": cur_hi, "seeds": seeds})
                cur_lo, cur_hi, seeds = lo, hi, [seed]
        windows.append({"source": source, "lo": cur_lo, "hi": cur_hi, "seeds": seeds})
    return windows


def expand_context(
    docs: list[dict],
    *,
    window: int = 1,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> list[dict]:
    if not docs:
        return []
    client = client or get_qdrant()
    collection = collection or get_settings().qdrant_collection

    if window <= 0:
        return [
            {
                "source": d.get("source", ""),
                "chunk_ids": [d.get("chunk_id")],
                "meta": d.get("meta", ""),
                "text": d.get("text", ""),
                "chunks": [{**d, "role": "hit"}],
                "rerank_score": d.get("rerank_score"),
                "rrf_score": d.get("rrf_score"),
                "score": d.get("score"),
            }
            for d in docs
        ]

    intervals = []
    for d in docs:
        cid = int(d.get("chunk_id"))
        intervals.append((d.get("source") or "", cid - window, cid + window, d))
    merged = _merge_intervals(intervals)

    def seed_score(s: dict) -> float:
        for key in ("rerank_score", "rrf_score", "score"):
            v = s.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        return -1.0

    merged.sort(key=lambda w: max((seed_score(s) for s in w["seeds"]), default=-1.0), reverse=True)

    result: list[dict] = []
    for w in merged:
        source, lo, hi = w["source"], w["lo"], w["hi"]
        hit_ids = {int(s.get("chunk_id")) for s in w["seeds"]}
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="source", match=qm.MatchValue(value=source)),
                    qm.FieldCondition(key="chunk_id", range=qm.Range(gte=lo, lte=hi)),
                ]
            ),
            limit=hi - lo + 1,
            with_payload=True,
            with_vectors=False,
        )
        chunks = []
        for p in sorted(points, key=lambda x: int((x.payload or {}).get("chunk_id", 0))):
            cid = int((p.payload or {}).get("chunk_id"))
            chunks.append(_payload_row(p, role="hit" if cid in hit_ids else "neighbor"))
        if not chunks:
            continue
        top_seed = max(w["seeds"], key=seed_score) if w["seeds"] else {}
        result.append(
            {
                "source": source,
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "meta": " | ".join(dict.fromkeys(c.get("meta") or "" for c in chunks if c.get("meta"))),
                "text": "\n\n".join(c["text"] for c in chunks if c.get("text")),
                "chunks": chunks,
                "rerank_score": top_seed.get("rerank_score"),
                "rrf_score": top_seed.get("rrf_score"),
                "score": top_seed.get("score"),
                "window": f"{lo}-{hi}",
            }
        )
    return result


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, d in enumerate(hits, 1):
        src = d.get("source") or ""
        meta = d.get("meta") or ""
        parts.append(f"[{i}] source={src} meta={meta}\n{d.get('text', '')}")
    return "\n\n---\n\n".join(parts) if parts else "（无检索结果）"
