"""Online RAG: rewrite → multi-recall → rerank → context expand → generate."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator

from app.online.context_expand import build_context, expand_context
from app.online.generation import generate_answer
from app.online.multi_recall import multi_recall
from app.online.query_rewrite import QueryPlan, plan_query
from app.online.reranking import rerank_hits
from app.rag_state import get_rag_runtime


@dataclass
class OnlineResult:
    query: str
    plan: QueryPlan
    path_queries: list[tuple[str, str]] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)
    context: str = ""


def run_online_retrieval(query: str, *, enable_rewrite: bool = True) -> OnlineResult:
    runtime = get_rag_runtime()
    if enable_rewrite:
        plan = plan_query(query)
    else:
        plan = QueryPlan(original=query.strip(), rewrite=query.strip())

    fused, path_queries = multi_recall(plan, recall_top=runtime.retrieve_top_k)
    pool = fused[: max(runtime.retrieve_top_k * 2, 20)]
    ranked = rerank_hits(query, pool, top_n=runtime.rerank_top_n)
    expanded = expand_context(ranked, window=runtime.context_window)
    return OnlineResult(
        query=query,
        plan=plan,
        path_queries=path_queries,
        hits=expanded,
        context=build_context(expanded),
    )


def run_rag_answer(query: str, *, stream: bool = False) -> str | Iterator[str]:
    runtime = get_rag_runtime()
    if not runtime.rag_enabled:
        llm_answer = generate_answer(query, [], stream=stream)
        return llm_answer
    result = run_online_retrieval(query)
    return generate_answer(query, result.hits, stream=stream)
