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

    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    closeai_api_key: str = ""
    closeai_base_url: str = "https://api.openai-proxy.org/v1"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embed_model: str = "Pro/BAAI/bge-m3"
    embed_dim: int = 1024

    milvus_uri: str = "http://localhost:19530"
    milvus_db: str = "funlab"
    milvus_collection: str = "funlab_docs"

    knowledge_dir: str = "./data/knowledge"
    mineru_api_token: str = ""

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "funlab-assistant"

    @property
    def knowledge_path(self) -> Path:
        path = Path(self.knowledge_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
