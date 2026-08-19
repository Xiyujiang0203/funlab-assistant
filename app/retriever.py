from app.embeddings import get_embed_model
from app.rag_state import get_rag_runtime
from app.reranker import rerank_hits
from app.vectorstore import VectorStore


class KnowledgeRetriever:
    def __init__(self):
        self.embed_model = get_embed_model()
        self.store = VectorStore()

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:
        runtime = get_rag_runtime()
        query_vector = self.embed_model.embed_query(query)
        hits = self.store.search(query_vector, top_k=max(top_k, runtime.retrieve_top_k))
        formatted = []
        for hit in hits:
            entity = hit.get("entity", {})
            score = hit.get("distance", 0.0)
            if score < score_threshold:
                continue
            formatted.append(
                {
                    "text": entity.get("text", ""),
                    "source": entity.get("source", "unknown"),
                    "chunk_id": entity.get("chunk_id", "unknown"),
                    "file_type": entity.get("file_type", "unknown"),
                    "score": score,
                }
            )
        if runtime.rag_enabled and formatted:
            try:
                formatted = rerank_hits(query, formatted)
            except Exception:
                pass
        return formatted[:top_k]

    @staticmethod
    def format_hits(hits: list[dict]) -> str:
        if not hits:
            return "未检索到相关内容。"
        blocks = [hit["text"].strip() for hit in hits if hit.get("text", "").strip()]
        if not blocks:
            return "未检索到相关内容。"
        return "\n\n".join(blocks)
