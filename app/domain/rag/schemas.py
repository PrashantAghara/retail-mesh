from pydantic import BaseModel


class Source(BaseModel):
    title: str
    category: str | None = None


class SupportQueryRequest(BaseModel):
    query: str


class SupportQueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]