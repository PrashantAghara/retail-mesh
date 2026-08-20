from fastapi import APIRouter, Depends

from app.api.deps import get_fulfillment_models
from app.domain.fulfillment.model_registry import FulfillmentModelsContainer
from app.domain.fulfillment.schemas import (
    FulfillmentQueryRequest,
    FulfillmentQueryResponse,
)

router = APIRouter(prefix="/fulfillment", tags=["fulfillment"])


@router.post("/query", response_model=FulfillmentQueryResponse)
def query_fulfillment(
    request: FulfillmentQueryRequest,
    models: FulfillmentModelsContainer = Depends(get_fulfillment_models),  # noqa: B008
):
    result = models.agent.invoke({"messages": [("user", request.query)]})
    answer = result["messages"][-1].content
    return FulfillmentQueryResponse(query=request.query, answer=answer)
