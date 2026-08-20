import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.fulfillment import router as fulfillment_router
from app.api.nlp import router as nlp_router
from app.api.rag import router as rag_router
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.domain.fulfillment.model_registry import load_fulfillment_models
from app.domain.nlp.model_registry import load_nlp_models
from app.domain.rag.model_registry import load_rag_models

logger = logging.getLogger(__name__)


async def _load_model(settings: Settings, name: str, loader: Callable) -> Awaitable:
    """Load a model off the event loop to avoid blocking startup."""
    logger.info("Loading %s models...", name)
    try:
        result = await asyncio.to_thread(loader, settings)
        logger.info("%s models loaded successfully", name)
        return result
    except Exception as e:
        logger.error("Failed to load %s models: %s", name, e)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown.
    
    Loads all ML models on startup and cleans up on shutdown.
    """
    settings = get_settings()

    app.state.nlp_models = await _load_model(settings, "NLP", load_nlp_models)
    app.state.rag_models = await _load_model(settings, "RAG", load_rag_models)
    app.state.fulfillment_models = await _load_model(
        settings, "fulfillment", load_fulfillment_models
    )

    yield

    logger.info("Shutting down...")


app = FastAPI(title="RetailMesh", lifespan=lifespan)
app.include_router(nlp_router)
app.include_router(rag_router)
app.include_router(fulfillment_router)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Translate application errors into consistent HTTP responses."""
    logger.error("AppError on %s %s: %s", request.method, request.url.path, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
