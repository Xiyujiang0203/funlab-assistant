from langchain.embeddings import init_embeddings

from app.config import get_settings


def get_embed_model():
    settings = get_settings()
    return init_embeddings(
        model=f"openai:{settings.embed_model}",
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
    )
