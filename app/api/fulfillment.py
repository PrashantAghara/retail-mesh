from fastapi import APIRouter, Depends

from app.api.deps import get_fulfillment_models
from app.fulfillment.model_registry import FulfillmentModels
from app.fulfillment.schemas import FulfillmentQueryRequest, FulfillmentQueryResponse

router = APIRouter(prefix="/fulfillment", tags=["fulfillment"])


@router.post("/query", response_model=FulfillmentQueryResponse)
def query_fulfillment(
    request: FulfillmentQueryRequest,
    models: FulfillmentModels = Depends(get_fulfillment_models),  # noqa: B008
):
    result = models.agent.invoke({"messages": [("user", request.query)]})
    answer = result["messages"][-1].content
    return {"query": request.query, "answer": answer}
