from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str
    hf_token: str | None = None
    database_url: str

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    intent_llm_model: str = "openai/gpt-oss-120b"
    intent_confidence_threshold: float = 0.65

    # RAG settings
    rag_collection_name: str = "retailmesh_knowledge_base"
    rag_bm25_k: int = 8
    rag_vector_k: int = 8
    rag_ensemble_weights: tuple[float, float] = (0.3, 0.7)
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_top_k: int = 3

    # Fulfillment settings
    fulfillment_order_id_prefix: str = "order_"
    fulfillment_order_id_length: int = 8
    fulfillment_max_product_results: int = 3

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
