from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"

    dashscope_api_key: str = ""
    embed_model: str = "text-embedding-v4"
    embed_dim: int = 1024
    rerank_model: str = "qwen3-rerank"

    qdrant_path: str = "./data/qdrant"
    qdrant_url: str = ""
    qdrant_collection: str = "funlab_chunks"

    knowledge_dir: str = "./data/knowledge"
    chunk_size: int = 800
    chunk_overlap: int = 120
    recall_top: int = 8
    rerank_top: int = 5
    rrf_pool: int = 20
    context_window: int = 1
    rag_enabled: bool = True

    @property
    def knowledge_path(self) -> Path:
        path = Path(self.knowledge_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def qdrant_store_path(self) -> Path:
        path = Path(self.qdrant_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
