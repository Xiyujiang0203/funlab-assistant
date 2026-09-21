"""Qdrant recall + Qwen3 rerank (DashScope qwen3-rerank)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from qdrant_client import QdrantClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QDRANT_PATH = ROOT / "test" / "4datebase" / "qdrant_data"
DEFAULT_OUT = HERE / "output"
DEFAULT_COLLECTION = "funlab_chunks"

EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBED_MODEL = "text-embedding-v4"
RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_MODEL = "qwen3-rerank"


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


def rerank_qwen(query: str, docs: list[dict], api_key: str, top_n: int) -> list[dict]:
    if not docs:
        return []
    documents = [d.get("text", "") for d in docs]
    resp = requests.post(
        RERANK_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": RERANK_MODEL,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": min(top_n, len(documents)), "return_documents": False},
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


def write_preview(path: Path, query: str, recalled: list[dict], ranked: list[dict]) -> None:
    lines = [
        f"# Rerank Preview",
        "",
        f"- query: {query}",
        f"- embed_model: `{EMBED_MODEL}`",
        f"- rerank_model: `{RERANK_MODEL}`",
        f"- recall_top: {len(recalled)}",
        f"- rerank_top: {len(ranked)}",
        "",
        "## After Rerank",
        "",
    ]
    for i, row in enumerate(ranked, 1):
        lines.append(f"### {i}. chunk_id={row.get('chunk_id')} · rerank={row.get('rerank_score'):.4f} · vec={row.get('score'):.4f}")
        lines.append("")
        lines.append(f"- source: `{row.get('source')}`")
        lines.append(f"- meta: {row.get('meta')}")
        lines.append("")
        lines.append("```markdown")
        lines.append(row.get("text", ""))
        lines.append("```")
        lines.append("")
    lines.extend(["## Before Rerank (vector recall)", ""])
    for i, row in enumerate(recalled, 1):
        lines.append(f"### {i}. chunk_id={row.get('chunk_id')} · vec={row.get('score'):.4f}")
        lines.append("")
        lines.append(f"- source: `{row.get('source')}`")
        lines.append("")
        lines.append("```markdown")
        lines.append(row.get("text", "")[:500])
        lines.append("```")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Qdrant recall + Qwen rerank")
    parser.add_argument("query", nargs="?", default="实验室如何请假和考勤？")
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--store-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--url", default="", help="Qdrant http url; empty = local path")
    parser.add_argument("--recall-top", type=int, default=10)
    parser.add_argument("--rerank-top", type=int, default=5)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.api_key:
        print("缺少 API key：传 --api-key 或设置 DASHSCOPE_API_KEY", file=sys.stderr)
        return 2

    vector = embed_query(args.query, args.api_key)
    client = QdrantClient(url=args.url) if args.url else QdrantClient(path=str(args.store_path.resolve()))
    try:
        recalled = search_qdrant(client, args.collection, vector, args.recall_top)
        ranked = rerank_qwen(args.query, recalled, args.api_key, args.rerank_top)
    finally:
        client.close()

    out = args.output_dir.resolve() / "rerank_preview.md"
    write_preview(out, args.query, recalled, ranked)
    print(json.dumps({"query": args.query, "recall": len(recalled), "rerank": len(ranked), "output": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
