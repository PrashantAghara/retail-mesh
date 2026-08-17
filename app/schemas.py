from typing import Literal

from pydantic import BaseModel


class Intent(BaseModel):
    category: Literal["support", "fulfillment", "vision"]
    confidence: float
    sentiment: str | None = None


class AgentState(BaseModel):
    query: str
    intent: Intent | None = None
    response: str | None = None
    metadata: dict = {}
