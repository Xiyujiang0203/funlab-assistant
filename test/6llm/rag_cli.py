"""RAG CLI: query_rewrite + multi_recall + Qwen rerank + DeepSeek Flash."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import requests
from openai import OpenAI
from qdrant_client import QdrantClient

from context_expand import expand_context
from multi_recall import multi_recall
from query_rewrite import QueryPlan, plan_query

ROOT = HERE.parents[1]
QDRANT_PATH = ROOT / "test" / "4datebase" / "qdrant_data"
DEFAULT_COLLECTION = "funlab_chunks"

RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_MODEL = "qwen3-rerank"

SYSTEM = (
    "你是 FUNLAB 实验室助手。只根据给定检索片段回答，用简洁中文。"
    "若片段不足以回答，明确说不知道，不要编造。"
)


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


def build_context(docs: list[dict]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.get("source") or ""
        meta = d.get("meta") or ""
        parts.append(f"[{i}] source={src} meta={meta}\n{d.get('text', '')}")
    return "\n\n---\n\n".join(parts)


def answer(client: OpenAI, model: str, query: str, docs: list[dict]) -> str:
    context = build_context(docs) if docs else "（无检索结果）"
    user = f"检索片段：\n{context}\n\n用户问题：{query}"
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        stream=True,
    )
    chunks: list[str] = []
    for event in stream:
        delta = event.choices[0].delta.content or ""
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()
    return "".join(chunks)


def rag_once(
    *,
    query: str,
    qdrant: QdrantClient,
    collection: str,
    dashscope_key: str,
    llm: OpenAI,
    llm_model: str,
    recall_top: int,
    rerank_top: int,
    show_sources: bool,
    enable_rewrite: bool,
    rrf_pool: int,
    context_window: int,
) -> None:
    if enable_rewrite:
        plan = plan_query(llm, llm_model, query)
    else:
        plan = QueryPlan(original=query.strip(), rewrite=query.strip())

    fused, path_queries = multi_recall(
        plan,
        qdrant=qdrant,
        collection=collection,
        api_key=dashscope_key,
        recall_top=recall_top,
    )
    pool = fused[:rrf_pool]
    ranked = rerank_qwen(query, pool, dashscope_key, rerank_top)
    expanded = expand_context(ranked, qdrant, collection, window=context_window)

    if show_sources:
        print("\n[Query 改写]")
        print(f"  原始: {plan.original}")
        print(f"  改写: {plan.rewrite}")
        print(f"  关键词: {plan.keywords}")
        print(f"  子问题: {plan.sub_queries}")
        print(f"[多路召回] {len(path_queries)} 路 → RRF {len(pool)} → 重排 {len(ranked)}")
        for name, q in path_queries:
            print(f"  · {name}: {q}")
        for i, d in enumerate(ranked, 1):
            rs = d.get("rerank_score")
            rs_s = f"{rs:.4f}" if isinstance(rs, (int, float)) else "-"
            rrf = d.get("rrf_score")
            rrf_s = f"{rrf:.4f}" if isinstance(rrf, (int, float)) else "-"
            print(f"  {i}. {d.get('source')}#{d.get('chunk_id')} rerank={rs_s} rrf={rrf_s}")
        print(f"[上下文扩展] window=±{context_window} → {len(expanded)} 段")
        for i, w in enumerate(expanded, 1):
            ids = w.get("chunk_ids") or []
            roles = ",".join(
                ("H" if c.get("role") == "hit" else "N") for c in (w.get("chunks") or [])
            )
            print(f"  {i}. {w.get('source')}#{ids} [{roles}]")
        print()
    print("助手: ", end="", flush=True)
    answer(llm, llm_model, query, expanded)


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="FUNLAB RAG CLI (DeepSeek Flash)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--store-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--url", default="", help="Qdrant http url; empty = local path")
    parser.add_argument("--recall-top", type=int, default=8)
    parser.add_argument("--rerank-top", type=int, default=5)
    parser.add_argument("--rrf-pool", type=int, default=20)
    parser.add_argument("--window", type=int, default=1, help="命中 chunk 前后各取几个邻居，0=不扩展")
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--show-sources", action="store_true", default=True)
    parser.add_argument("--hide-sources", action="store_true")
    parser.add_argument("--deepseek-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--deepseek-base", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))
    parser.add_argument("--dashscope-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("query", nargs="?", default="", help="单次提问；省略则进入交互")
    args = parser.parse_args()

    show_sources = not args.hide_sources
    if not args.deepseek_key:
        print("缺少 DEEPSEEK_API_KEY", file=sys.stderr)
        return 2
    if not args.dashscope_key:
        print("缺少 DASHSCOPE_API_KEY（embed/rerank）", file=sys.stderr)
        return 2

    llm = OpenAI(api_key=args.deepseek_key, base_url=args.deepseek_base)
    qdrant = QdrantClient(url=args.url) if args.url else QdrantClient(path=str(args.store_path.resolve()))

    kwargs = dict(
        qdrant=qdrant,
        collection=args.collection,
        dashscope_key=args.dashscope_key,
        llm=llm,
        llm_model=args.model,
        recall_top=args.recall_top,
        rerank_top=args.rerank_top,
        show_sources=show_sources,
        enable_rewrite=not args.no_rewrite,
        rrf_pool=args.rrf_pool,
        context_window=args.window,
    )

    try:
        if args.query.strip():
            rag_once(query=args.query.strip(), **kwargs)
            return 0

        print("FUNLAB RAG  |  rewrite + 多路召回 + 上下文扩展  |  exit/quit  |  /sources")
        while True:
            try:
                q = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                break
            if q.lower() in {"/sources", "sources"}:
                show_sources = not show_sources
                kwargs["show_sources"] = show_sources
                print(f"来源显示: {'开' if show_sources else '关'}")
                continue
            try:
                rag_once(query=q, **kwargs)
            except Exception as e:
                print(f"错误: {e}", file=sys.stderr)
    finally:
        qdrant.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
