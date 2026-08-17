from fastapi import APIRouter, Depends

from app.api.deps import get_nlp_models
from app.core.config import Settings, get_settings
from app.nlp.model_loaders import NLPModels
from app.nlp.models import AnalyzeRequest, AnalyzeResponse
from app.nlp.service import get_intent

router = APIRouter(prefix="/nlp", tags=["nlp"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    models: NLPModels = Depends(get_nlp_models),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
):
    return get_intent(request.query, models, settings)
