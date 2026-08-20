from fastapi import Request

from app.core.config import get_settings
from app.nlp.model_loaders import NLPModels
from app.rag.model_registry import RAGModels


def get_nlp_models(request: Request) -> NLPModels:
    return request.app.state.nlp_models


def get_rag_models(request: Request) -> RAGModels:
    return request.app.state.rag_models


SettingsDep = get_settings
NLPModelsDep = get_nlp_models
RagModelsDep = get_rag_models
