from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, ValidationError as PydanticValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str
    hf_token: str | None = None
    database_url: str

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    intent_llm_model: str = "openai/gpt-oss-120b"
    intent_confidence_threshold: float = 0.65
    vision_weights_path: str = "models/shelf_detection_best.pt"

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

    @field_validator("intent_confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("intent_confidence_threshold must be between 0 and 1")
        return v

    @field_validator("rag_ensemble_weights")
    @classmethod
    def validate_ensemble_weights(cls, v: tuple[float, float]) -> tuple[float, float]:
        if abs(sum(v) - 1.0) > 0.001:
            raise ValueError("rag_ensemble_weights must sum to 1.0")
        return v

    @field_validator("vision_weights_path")
    @classmethod
    def validate_vision_weights_path(cls, v: str) -> str:
        path = Path(v)
        if not path.is_absolute():
            # Allow relative paths but warn they're relative to cwd
            pass
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


def validate_startup_config() -> Settings:
    """Validate all required configuration at startup, fail fast if invalid."""
    try:
        settings = get_settings()

        # Check required API keys
        if not settings.groq_api_key or not settings.groq_api_key.strip():
            raise ValueError("GROQ_API_KEY is required but not set")

        # Check database URL format
        if not settings.database_url or not settings.database_url.startswith(("postgresql://", "postgresql+psycopg2://")):
            raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string")

        # Check vision weights file exists
        weights_path = Path(settings.vision_weights_path)
        if not weights_path.exists():
            raise ValueError(f"Vision weights file not found: {settings.vision_weights_path}")

        return settings
    except PydanticValidationError as e:
        raise ValueError(f"Configuration validation failed: {e}") from e
