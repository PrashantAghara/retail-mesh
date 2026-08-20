from typing import Literal

from pydantic import BaseModel, Field


class IntentOutput(BaseModel):
    category: Literal["support", "fulfillment", "vision"] = Field(
        description="Which agent should handle this query"
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Sentiment of the query"
    )
    reasoning: str = Field(description="Brief reason for the classification")


class AnalyzeRequest(BaseModel):
    query: str


class AnalyzeResponse(BaseModel):
    query: str
    category: str
    confidence: float | None = None
    sentiment: str
    method: Literal["embedding", "llm_fallback"]
    reasoning: str | None = None


class Intent(BaseModel):
    category: Literal["support", "fulfillment", "vision"]
    confidence: float
    sentiment: str | None = None


class AgentState(BaseModel):
    query: str
    intent: Intent | None = None
    response: str | None = None
    metadata: dict = {}