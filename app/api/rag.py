from fastapi import APIRouter, Depends

from app.api.deps import get_rag_models
from app.rag.model_registry import RAGModels
from app.rag.models import SupportQueryRequest, SupportQueryResponse
from app.rag.service import support_agent

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/support", response_model=SupportQueryResponse)
def query_support(
    request: SupportQueryRequest,
    models: RAGModels = Depends(get_rag_models),  # noqa: B008
):
    return support_agent(request.query, models)
