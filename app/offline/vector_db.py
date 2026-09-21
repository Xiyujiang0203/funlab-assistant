"""Qdrant upsert (from test/4datebase)."""
from __future__ import annotations

import uuid

from qdrant_client.http import models as qm

from app.config import get_settings
from app.qdrant_client import ensure_collection, get_qdrant


def stable_id(source: str, chunk_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{chunk_id}"))


def store_vectors(rows: list[dict], *, rebuild: bool = False) -> int:
    if not rows:
        return 0
    dim = int(rows[0].get("dimensions") or len(rows[0]["embedding"]))
    name = ensure_collection(dim, recreate=rebuild)
    client = get_qdrant()
    points = []
    for row in rows:
        vec = row["embedding"]
        points.append(
            qm.PointStruct(
                id=stable_id(row.get("source", ""), int(row["chunk_id"])),
                vector=vec,
                payload={
                    "source": row.get("source", ""),
                    "chunk_id": row.get("chunk_id"),
                    "chars": row.get("chars"),
                    "meta": row.get("meta", ""),
                    "text": row.get("text", ""),
                    "model": row.get("model", ""),
                    "dimensions": row.get("dimensions") or len(vec),
                },
            )
        )
    batch = 64
    for i in range(0, len(points), batch):
        client.upsert(collection_name=name, points=points[i : i + batch])
    return len(points)


def store_text_chunks(chunks: list[str], *, source: str = "manual", rebuild: bool = False) -> int:
    from app.offline.vectorization import vectorize_chunks

    rows = [
        {"source": source, "chunk_id": i, "chars": len(t), "meta": "", "text": t}
        for i, t in enumerate(chunks, 1)
        if t.strip()
    ]
    embedded = vectorize_chunks(rows)
    return store_vectors(embedded, rebuild=rebuild)
