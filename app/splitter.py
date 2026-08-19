from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag_state import get_rag_runtime


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    runtime = get_rag_runtime()
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or runtime.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else runtime.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
    )
