from fastapi import Request

from app.core.config import get_settings
from app.domain.fulfillment.model_registry import FulfillmentModelsContainer
from app.domain.nlp.model_registry import NLPModelsContainer
from app.domain.rag.model_registry import RAGModelsContainer


def get_nlp_models(request: Request) -> NLPModelsContainer:
    return request.app.state.nlp_models


def get_rag_models(request: Request) -> RAGModelsContainer:
    return request.app.state.rag_models


def get_fulfillment_models(request: Request) -> FulfillmentModelsContainer:
    return request.app.state.fulfillment_models


SettingsDep = get_settings
NLPModelsDep = get_nlp_models
RagModelsDep = get_rag_models
FulfillmentModelsDep = get_fulfillment_models
