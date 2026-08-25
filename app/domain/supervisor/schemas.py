from typing import Optional, TypedDict

from pydantic import BaseModel


class SupervisorState(TypedDict):
    query: str
    image_path: Optional[str]
    category: Optional[str]
    confidence: Optional[float]
    method: Optional[str]
    response: Optional[str]
    needs_image: bool


class SupervisorRequest(BaseModel):
    query: str
    image_path: Optional[str] = None


class SupervisorResponse(BaseModel):
    query: str
    category: Optional[str]
    response: str
    needs_image: bool
