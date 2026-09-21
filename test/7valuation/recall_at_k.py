"""Recall@K evaluation against Qdrant (optional Qwen rerank)."""
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
DEFAULT_EVAL = HERE / "eval_set.jsonl"
DEFAULT_COLLECTION = "funlab_chunks"
DEFAULT_KS = (1, 3, 5, 10)

EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBED_MODEL = "text-embedding-v4"
RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_MODEL = "qwen3-rerank"


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def load_eval(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
                "score": float(h.score or 0.0),
                "source": Path(str(payload.get("source", ""))).name,
                "chunk_id": int(payload.get("chunk_id")),
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


def norm_doc(name: str) -> str:
    name = Path(name).name
    if name.endswith(".chunks.md"):
        return name[: -len(".chunks.md")] + ".md"
    return name


def hit(retrieved: list[dict], gold_doc: str, gold_ids: set[int]) -> bool:
    gold = norm_doc(gold_doc)
    for row in retrieved:
        if norm_doc(row["source"]) == gold and row["chunk_id"] in gold_ids:
            return True
    return False


def recall_at_k(retrieved: list[dict], gold_doc: str, gold_ids: set[int], k: int) -> float:
    return 1.0 if hit(retrieved[:k], gold_doc, gold_ids) else 0.0


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Recall@K on eval_set.jsonl")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--store-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--url", default="")
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--recall-top", type=int, default=20)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("-o", "--output", type=Path, default=HERE / "output" / "recall_at_k.json")
    args = parser.parse_args()

    if not args.api_key:
        print("缺少 DASHSCOPE_API_KEY", file=sys.stderr)
        return 2

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    max_k = max(ks)
    fetch_n = max(args.recall_top, max_k)
    cases = load_eval(args.eval.resolve())
    if not cases:
        print("评估集为空", file=sys.stderr)
        return 1

    client = QdrantClient(url=args.url) if args.url else QdrantClient(path=str(args.store_path.resolve()))
    details = []
    sums = {k: 0.0 for k in ks}

    try:
        for case in cases:
            q = case["query"]
            gold_doc = case["doc"]
            gold_ids = {int(x) for x in case["gold_chunk_ids"]}
            vector = embed_query(q, args.api_key)
            recalled = search_qdrant(client, args.collection, vector, fetch_n)
            ranked = rerank_qwen(q, recalled, args.api_key, fetch_n) if args.rerank else recalled
            row = {
                "id": case.get("id"),
                "query": q,
                "doc": gold_doc,
                "gold_chunk_ids": sorted(gold_ids),
                "hits": {},
                "top": [
                    {"source": d["source"], "chunk_id": d["chunk_id"], "score": d.get("rerank_score", d["score"])}
                    for d in ranked[:max_k]
                ],
            }
            for k in ks:
                v = recall_at_k(ranked, gold_doc, gold_ids, k)
                row["hits"][f"recall@{k}"] = v
                sums[k] += v
            details.append(row)
            mark = "Y" if row["hits"][f"recall@{max_k}"] else "N"
            print(f"[{mark}] {case.get('id')}: {q[:40]}")
    finally:
        client.close()

    n = len(details)
    metrics = {f"recall@{k}": round(sums[k] / n, 4) for k in ks}
    out = {
        "n": n,
        "rerank": bool(args.rerank),
        "ks": list(ks),
        "metrics": metrics,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "n": n, "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
