import requests

from app.config import get_settings
from app.rag_state import get_rag_runtime


def rerank_hits(query: str, hits: list[dict]) -> list[dict]:
    if not query.strip() or len(hits) <= 1:
        return hits

    settings = get_settings()
    runtime = get_rag_runtime()

    if not settings.siliconflow_api_key or not runtime.rerank_model:
        return hits

    documents = [hit.get("text", "") for hit in hits]
    if not any(documents):
        return hits

    url = f"{settings.siliconflow_base_url.rstrip('/')}/rerank"
    payload = {
        "model": runtime.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": min(runtime.rerank_top_n, len(documents)),
        "return_documents": False,
        "max_chunks_per_doc": 1024,
        "overlap_tokens": 80,
    }
    headers = {
        "Authorization": f"Bearer {settings.siliconflow_api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()

    ranked: list[dict] = []
    for item in data.get("results", []):
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(hits):
            ranked.append(
                {
                    **hits[idx],
                    "rerank_score": item.get("relevance_score", item.get("score")),
                }
            )

    if not ranked:
        return hits

    leftovers = [hit for i, hit in enumerate(hits) if i not in {item.get("index") for item in data.get("results", [])}]
    return ranked + leftovers

