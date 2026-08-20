from pydantic import BaseModel


class FulfillmentQueryRequest(BaseModel):
    query: str


class FulfillmentQueryResponse(BaseModel):
    query: str
    answer: str
