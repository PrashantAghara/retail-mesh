from fastapi import Request

from app.core.config import get_settings
from app.nlp.model_loaders import NLPModels


def get_nlp_models(request: Request) -> NLPModels:
    return request.app.state.nlp_models


SettingsDep = get_settings
NLPModelsDep = get_nlp_models
