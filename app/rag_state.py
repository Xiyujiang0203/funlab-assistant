from dataclasses import asdict, dataclass

from app.config import get_settings


@dataclass
class RagRuntimeConfig:
    rag_enabled: bool = True
    embed_model: str = "Pro/BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    chunk_size: int = 300
    chunk_overlap: int = 60
    retrieve_top_k: int = 8
    rerank_top_n: int = 5


_runtime: RagRuntimeConfig | None = None


def get_rag_runtime() -> RagRuntimeConfig:
    global _runtime
    if _runtime is None:
        settings = get_settings()
        _runtime = RagRuntimeConfig(
            embed_model=settings.embed_model,
        )
    return _runtime


def update_rag_runtime(**kwargs) -> RagRuntimeConfig:
    runtime = get_rag_runtime()
    for key, value in kwargs.items():
        if value is None or not hasattr(runtime, key):
            continue
        setattr(runtime, key, value)

    if runtime.chunk_overlap >= runtime.chunk_size:
        runtime.chunk_overlap = max(0, runtime.chunk_size // 5)

    if runtime.rerank_top_n > runtime.retrieve_top_k:
        runtime.rerank_top_n = runtime.retrieve_top_k

    return runtime


def dump_rag_runtime() -> dict:
    return asdict(get_rag_runtime())

