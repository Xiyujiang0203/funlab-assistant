"""Upsert embedding jsonl into Qdrant (local path by default)."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_INPUT = ROOT / "test" / "3embedding" / "output"
DEFAULT_STORE = HERE / "qdrant_data"
DEFAULT_COLLECTION = "funlab_chunks"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_id(source: str, chunk_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{chunk_id}"))


def get_client(url: str | None, store_path: Path) -> QdrantClient:
    if url:
        return QdrantClient(url=url)
    store_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(store_path))


def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool) -> None:
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )


def upsert_rows(client: QdrantClient, collection: str, rows: list[dict]) -> int:
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
        client.upsert(collection_name=collection, points=points[i : i + batch])
    return len(points)


def main() -> int:
    parser = argparse.ArgumentParser(description="Store embeddings into Qdrant")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--store-path", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--url", default="", help="e.g. http://127.0.0.1:6333; empty = local path mode")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    src = args.input.resolve()
    files = [src] if src.is_file() else sorted(src.glob("*.embeddings.jsonl"))
    if not files:
        print(f"未找到 *.embeddings.jsonl：{src}", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    for path in files:
        rows = load_jsonl(path)
        all_rows.extend(rows)
        print(f"loaded {len(rows)} from {path.name}")

    if not all_rows:
        print("无数据", file=sys.stderr)
        return 1

    dim = int(all_rows[0].get("dimensions") or len(all_rows[0]["embedding"]))
    client = get_client(args.url or None, args.store_path.resolve())
    try:
        ensure_collection(client, args.collection, dim, args.recreate)
        n = upsert_rows(client, args.collection, all_rows)
        info = client.get_collection(args.collection)
        print(
            json.dumps(
                {
                    "collection": args.collection,
                    "upserted": n,
                    "points_count": info.points_count,
                    "vector_size": dim,
                    "store": args.url or str(args.store_path.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
