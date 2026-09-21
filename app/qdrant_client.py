from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings

_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is not None:
        return _client
    s = get_settings()
    if s.qdrant_url:
        _client = QdrantClient(url=s.qdrant_url)
    else:
        path = s.qdrant_store_path
        path.mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=str(path))
    return _client


def close_qdrant() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def ensure_collection(dim: int, recreate: bool = False) -> str:
    s = get_settings()
    client = get_qdrant()
    name = s.qdrant_collection
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
    return name
