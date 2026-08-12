from pymilvus import MilvusClient

from app.config import get_settings


class VectorStore:
    def __init__(self):
        settings = get_settings()
        self.uri = settings.milvus_uri
        self.db_name = settings.milvus_db
        self.collection_name = settings.milvus_collection
        self.embed_dim = settings.embed_dim
        self.client = MilvusClient(self.uri)

    def ensure_ready(self, rebuild: bool = False) -> None:
        dbs = self.client.list_databases()
        if self.db_name not in dbs:
            self.client.create_database(db_name=self.db_name)
        self.client.use_database(db_name=self.db_name)

        if rebuild and self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

        if not self.client.has_collection(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embed_dim,
                metric_type="COSINE",
            )

    def upsert_chunks(self, chunks: list[dict]) -> int:
        self.ensure_ready(rebuild=False)
        if not chunks:
            return 0
        self.client.upsert(collection_name=self.collection_name, data=chunks)
        self.client.flush(collection_name=self.collection_name)
        return len(chunks)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        self.ensure_ready(rebuild=False)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "chunk_id", "source", "file_type"],
        )
        return results[0] if results else []

    def count(self) -> int:
        self.ensure_ready(rebuild=False)
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return int(stats.get("row_count", 0))

    def ping(self) -> bool:
        try:
            self.client.list_databases()
            return True
        except Exception:
            return False
