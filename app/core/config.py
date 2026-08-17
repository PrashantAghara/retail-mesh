from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str
    hf_token: str | None = None
    database_url: str

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    intent_llm_model: str = "openai/gpt-oss-120b"
    intent_confidence_threshold: float = 0.65

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
