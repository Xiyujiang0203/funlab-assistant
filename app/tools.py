from langchain_core.tools import tool

from app.rag_state import get_rag_runtime
from app.retriever import KnowledgeRetriever

_retriever: KnowledgeRetriever | None = None


def _get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever


@tool
def search_funlab_knowledge(query: str) -> str:
    """搜索 Funlab 实验室知识库，用于回答实验室制度、项目、设备、流程、成员等相关问题。
    若返回"未检索到相关内容"，说明知识库中确实没有此信息，请直接告知用户，不要重复调用本工具。"""
    try:
        if not get_rag_runtime().rag_enabled:
            return "RAG 当前已关闭。"
        hits = _get_retriever().search(query, top_k=5)
        return KnowledgeRetriever.format_hits(hits)
    except Exception as exc:
        return f"知识库检索失败: {exc}"


FUNLAB_TOOLS = [search_funlab_knowledge]
