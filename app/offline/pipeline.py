"""Offline ingest pipeline: load → chunk → embed → qdrant."""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.offline.chunking import split_documents
from app.offline.document_processing import load_documents
from app.offline.vector_db import store_vectors
from app.offline.vectorization import vectorize_chunks
from app.rag_state import get_rag_runtime


def run_offline_ingest(rebuild: bool = False, file_paths: list[Path] | None = None) -> int:
    documents = load_documents(file_paths)
    if not documents:
        return 0
    runtime = get_rag_runtime()
    chunks = split_documents(
        documents,
        chunk_size=runtime.chunk_size,
        chunk_overlap=runtime.chunk_overlap,
    )
    if not chunks:
        return 0
    rows = vectorize_chunks(chunks)
    return store_vectors(rows, rebuild=rebuild)


def seed_from_test_ocr(rebuild: bool = True) -> int:
    """Import test/1pdf2markdown/1.3ocr markdown into knowledge + qdrant."""
    from app.config import PROJECT_ROOT

    ocr_dir = PROJECT_ROOT / "test" / "1pdf2markdown" / "1.3ocr"
    knowledge = get_settings().knowledge_path
    knowledge.mkdir(parents=True, exist_ok=True)
    paths = []
    for md in sorted(ocr_dir.glob("*.md")):
        target = knowledge / md.name
        target.write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
        paths.append(target)
    return run_offline_ingest(rebuild=rebuild, file_paths=paths)
