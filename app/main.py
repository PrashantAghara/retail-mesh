import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat_stream import router as chat_stream_router
from app.api.fulfillment import router as fulfillment_router
from app.api.nlp import router as nlp_router
from app.api.rag import router as rag_router
from app.api.vision import router as vision_router
from app.api.voice import router as voice_router
from app.core.config import Settings, validate_startup_config
from app.core.exceptions import AppError
from app.core.health import HealthStatus, run_health_checks
from app.core.logging import configure_logging
from app.core.models import load_shared_models
from app.domain.fulfillment.model_registry import load_fulfillment_models
from app.domain.nlp.model_registry import load_nlp_models
from app.domain.rag.model_registry import load_rag_models
from app.domain.supervisor.model_registry import load_supervisor_models
from app.domain.vision.model_registry import load_vision_models
from app.domain.voice.model_registry import load_voice_models

# Validate configuration at startup - fail fast if invalid
settings = validate_startup_config()
configure_logging(json_format=os.getenv("LOG_JSON", "false").lower() == "true")
logger = logging.getLogger(__name__)

load_dotenv()


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

    Loads shared models once, then domain-specific models concurrently,
    reusing the shared embedder/LLM instead of duplicating them.
    """
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    shared = await _load_model(settings, "shared", load_shared_models)

    (
        nlp_models,
        rag_models,
        fulfillment_models,
        vision_models,
        voice_models,
    ) = await asyncio.gather(
        _load_model(settings, "NLP", lambda s: load_nlp_models(s, shared)),
        _load_model(settings, "RAG", lambda s: load_rag_models(s, shared)),
        _load_model(
            settings, "fulfillment", lambda s: load_fulfillment_models(s, shared)
        ),
        _load_model(settings, "vision", lambda s: load_vision_models(s, shared)),
        _load_model(settings, "voice", load_voice_models),
    )
    supervisor_models = await _load_model(
        settings,
        "supervisor",
        lambda s: load_supervisor_models(
            s, shared, nlp_models, rag_models, fulfillment_models, vision_models
        ),
    )
    app.state.supervisor_models = supervisor_models
    app.state.nlp_models = nlp_models
    app.state.rag_models = rag_models
    app.state.fulfillment_models = fulfillment_models
    app.state.vision_models = vision_models
    app.state.voice_models = voice_models

    yield

    logger.info("Shutting down...")


app = FastAPI(title="RetailMesh", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(nlp_router)
app.include_router(rag_router)
app.include_router(fulfillment_router)
app.include_router(vision_router)
app.include_router(voice_router)
app.include_router(chat_stream_router)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Translate application errors into consistent HTTP responses."""
    logger.error(
        "AppError on %s %s: %s (code=%s)",
        request.method,
        request.url.path,
        exc.message,
        exc.error_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic/FastAPI validation errors."""
    logger.warning(
        "Validation error on %s %s: %s", request.method, request.url.path, exc.errors()
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
            }
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected errors."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {},
            }
        },
    )


@app.get("/health")
def health(request: Request):
    """Comprehensive health check endpoint."""
    response = run_health_checks(dict(request.app.state))
    status_code = 200 if response.status == HealthStatus.HEALTHY else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": response.status.value,
            "version": response.version,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": c.latency_ms,
                    "details": c.details,
                    "error": c.error,
                }
                for c in (response.components or [])
            ],
        },
    )
