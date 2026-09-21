"""DashScope embedding (from test/3embedding)."""
from __future__ import annotations

import time

import requests

from app.config import get_settings

EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"


def embed_texts(texts: list[str], *, batch_size: int = 10) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    if not s.dashscope_api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY")
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            EMBED_URL,
            headers={
                "Authorization": f"Bearer {s.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": s.embed_model,
                "input": batch,
                "encoding_format": "float",
                "dimensions": s.embed_dim,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"embed failed {resp.status_code}: {resp.text[:300]}")
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        vectors.extend(row["embedding"] for row in data)
        time.sleep(0.05)
    return vectors


def vectorize_chunks(chunks: list[dict]) -> list[dict]:
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    s = get_settings()
    out = []
    for chunk, vec in zip(chunks, vectors):
        out.append(
            {
                **chunk,
                "embedding": vec,
                "model": s.embed_model,
                "dimensions": len(vec),
            }
        )
    return out
