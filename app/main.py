from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.nlp import router as nlp_router
from app.core.config import get_settings
from app.nlp.model_loaders import load_nlp_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.nlp_models = load_nlp_models(settings)
    yield


app = FastAPI(title="RetailMesh", lifespan=lifespan)
app.include_router(nlp_router)


@app.get("/health")
def health():
    return {"status": "ok"}
