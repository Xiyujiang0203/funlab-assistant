"""③ 上下文窗口扩展：命中 chunk 后带上前后相邻 chunk，保证上下文完整。"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def _payload_row(point, *, role: str, hit_score: float | None = None) -> dict:
    payload = point.payload or {}
    return {
        "id": str(point.id),
        "source": payload.get("source", ""),
        "chunk_id": int(payload.get("chunk_id")),
        "meta": payload.get("meta", ""),
        "text": payload.get("text", ""),
        "role": role,
        "hit_score": hit_score,
    }


def fetch_window(
    client: QdrantClient,
    collection: str,
    source: str,
    chunk_id: int,
    window: int = 1,
) -> list[dict]:
    """取同一 source 下 [chunk_id-window, chunk_id+window] 的 chunk，按 id 排序。"""
    if window < 0:
        window = 0
    lo, hi = chunk_id - window, chunk_id + window
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qm.Filter(
            must=[
                qm.FieldCondition(key="source", match=qm.MatchValue(value=source)),
                qm.FieldCondition(
                    key="chunk_id",
                    range=qm.Range(gte=lo, lte=hi),
                ),
            ]
        ),
        limit=max(2 * window + 1, 1),
        with_payload=True,
        with_vectors=False,
    )
    rows = []
    for p in points:
        cid = int((p.payload or {}).get("chunk_id"))
        role = "hit" if cid == chunk_id else "neighbor"
        rows.append(_payload_row(p, role=role))
    rows.sort(key=lambda r: int(r["chunk_id"]))
    return rows


def _merge_intervals(intervals: list[tuple[str, int, int, dict]]) -> list[dict]:
    """intervals: (source, lo, hi, seed_doc). 合并同 source 重叠区间。"""
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
    client: QdrantClient,
    collection: str,
    *,
    window: int = 1,
) -> list[dict]:
    """
    对每个命中 chunk 扩展前后 window 个邻居，重叠区间合并。
    返回若干 context window，每项含合并后的 text 与组成 chunks。
    """
    if not docs:
        return []
    if window <= 0:
        # 不扩展：原样包装
        out = []
        for d in docs:
            out.append(
                {
                    "source": d.get("source", ""),
                    "chunk_ids": [d.get("chunk_id")],
                    "meta": d.get("meta", ""),
                    "text": d.get("text", ""),
                    "chunks": [{**d, "role": "hit"}],
                    "hit": True,
                    "rerank_score": d.get("rerank_score"),
                    "rrf_score": d.get("rrf_score"),
                    "score": d.get("score"),
                }
            )
        return out

    intervals = []
    for d in docs:
        src = d.get("source") or ""
        cid = int(d.get("chunk_id"))
        intervals.append((src, cid - window, cid + window, d))

    merged = _merge_intervals(intervals)
    def seed_score(s: dict) -> float:
        best = -1.0
        for key in ("rerank_score", "rrf_score", "score"):
            v = s.get(key)
            if isinstance(v, (int, float)):
                best = max(best, float(v))
                break
        return best

    def window_rank(w: dict) -> float:
        return max((seed_score(s) for s in w["seeds"]), default=-1.0)

    merged.sort(key=window_rank, reverse=True)

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
            role = "hit" if cid in hit_ids else "neighbor"
            chunks.append(_payload_row(p, role=role))
        if not chunks:
            continue
        text = "\n\n".join(c["text"] for c in chunks if c.get("text"))
        metas = " | ".join(dict.fromkeys(c.get("meta") or "" for c in chunks if c.get("meta")))
        top_seed = max(w["seeds"], key=seed_score) if w["seeds"] else {}
        result.append(
            {
                "source": source,
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "meta": metas,
                "text": text,
                "chunks": chunks,
                "hit": True,
                "rerank_score": top_seed.get("rerank_score"),
                "rrf_score": top_seed.get("rrf_score"),
                "score": top_seed.get("score"),
                "window": f"{lo}-{hi}",
            }
        )
    return result
