from fastapi import APIRouter, Depends

from app.api.deps import get_rag_models
from app.domain.rag.model_registry import RAGModelsContainer
from app.domain.rag.schemas import SupportQueryRequest, SupportQueryResponse
from app.domain.rag.service import support_agent

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/support", response_model=SupportQueryResponse)
def query_support(
    request: SupportQueryRequest,
    models: RAGModelsContainer = Depends(get_rag_models),  # noqa: B008
):
    return support_agent(request.query, models)
