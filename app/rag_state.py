"""Runtime RAG toggles (in-memory)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from app.config import get_settings


@dataclass
class RagRuntime:
    rag_enabled: bool = True
    embed_model: str = "text-embedding-v4"
    rerank_model: str = "qwen3-rerank"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieve_top_k: int = 8
    rerank_top_n: int = 5
    context_window: int = 1


_runtime: RagRuntime | None = None


def get_rag_runtime() -> RagRuntime:
    global _runtime
    if _runtime is None:
        s = get_settings()
        _runtime = RagRuntime(
            rag_enabled=s.rag_enabled,
            embed_model=s.embed_model,
            rerank_model=s.rerank_model,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            retrieve_top_k=s.recall_top,
            rerank_top_n=s.rerank_top,
            context_window=s.context_window,
        )
    return _runtime


def update_rag_runtime(**kwargs) -> RagRuntime:
    runtime = get_rag_runtime()
    for k, v in kwargs.items():
        if hasattr(runtime, k) and v is not None:
            setattr(runtime, k, v)
    return runtime


def dump_rag_runtime() -> dict:
    return asdict(get_rag_runtime())
