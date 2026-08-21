from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import Settings
from app.domain.nlp.schemas import IntentOutput


@runtime_checkable
class SentimentAnalyzer(Protocol):
    def __call__(self, text: str) -> list[dict]: ...


@runtime_checkable
class IntentChain(Protocol):
    def invoke(self, input: dict) -> IntentOutput: ...


@runtime_checkable
class SupportChain(Protocol):
    def invoke(self, input: dict) -> Any: ...


@runtime_checkable
class FulfillmentAgent(Protocol):
    def invoke(self, input: dict) -> Any: ...


@dataclass
class SharedModels:
    embedder: HuggingFaceEmbeddings
    llm: ChatGroq


def load_shared_models(settings: Settings) -> SharedModels:
    """Models genuinely reused across domains — loaded exactly once."""
    embedder = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.intent_llm_model)
    return SharedModels(embedder=embedder, llm=llm)
