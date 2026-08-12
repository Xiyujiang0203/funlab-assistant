from pathlib import Path

from app.embeddings import get_embed_model
from app.loaders import load_file, load_knowledge_dir
from app.splitter import get_text_splitter
from app.vectorstore import VectorStore


def ingest(rebuild: bool = False, file_paths: list[Path] | None = None) -> int:
    if file_paths:
        documents = []
        for fp in file_paths:
            documents.extend(load_file(Path(fp)))
    else:
        documents = load_knowledge_dir()

    if not documents:
        return 0

    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    embed_model = get_embed_model()
    store = VectorStore()
    store.ensure_ready(rebuild=rebuild)

    start_id = 0 if rebuild else store.count()
    texts = [c.page_content for c in chunks]
    vectors = embed_model.embed_documents(texts)

    data = []
    for i, chunk in enumerate(chunks):
        data.append(
            {
                "id": start_id + i,
                "vector": vectors[i],
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_id": start_id + i,
                "file_type": chunk.metadata.get("file_type", "unknown"),
            }
        )

    return store.upsert_chunks(data)
