from typing import Any, Protocol, runtime_checkable

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