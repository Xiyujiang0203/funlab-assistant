from langchain.embeddings import init_embeddings

from app.config import get_settings
from app.rag_state import get_rag_runtime


def get_embed_model():
    settings = get_settings()
    runtime = get_rag_runtime()
    return init_embeddings(
        model=f"openai:{runtime.embed_model or settings.embed_model}",
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
    )
