import logging
import time
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import get_engine

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    latency_ms: float | None = None
    details: str | None = None
    error: str | None = None


@dataclass
class HealthResponse:
    status: HealthStatus
    version: str = "0.1.0"
    components: list[ComponentHealth] | None = None


def check_database() -> ComponentHealth:
    """Check database connectivity."""
    start = time.perf_counter()
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error("Database health check failed: %s", e)
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency_ms, 2),
            error=str(e),
        )


def check_vectorstore() -> ComponentHealth:
    """Check vector store connectivity."""
    start = time.perf_counter()
    try:
        settings = get_settings()
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_postgres import PGVector

        embedder = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        database_url = settings.database_url.replace(
            "postgresql://", "postgresql+psycopg2://"
        )
        vectorstore = PGVector(
            embeddings=embedder,
            collection_name=settings.rag_collection_name,
            connection=database_url,
        )
        # Try a simple similarity search
        vectorstore.similarity_search("test", k=1)
        latency_ms = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="vectorstore",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error("Vector store health check failed: %s", e)
        return ComponentHealth(
            name="vectorstore",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency_ms, 2),
            error=str(e),
        )


def check_models_loaded(app_state: dict) -> ComponentHealth:
    """Check if all required models are loaded in app state."""
    required_models = [
        "supervisor_models",
        "nlp_models",
        "rag_models",
        "fulfillment_models",
        "vision_models",
        "voice_models",
    ]
    missing = [m for m in required_models if m not in app_state or app_state[m] is None]

    if missing:
        return ComponentHealth(
            name="models",
            status=HealthStatus.UNHEALTHY,
            error=f"Missing models: {', '.join(missing)}",
        )

    return ComponentHealth(
        name="models",
        status=HealthStatus.HEALTHY,
    )


def get_overall_status(components: list[ComponentHealth]) -> HealthStatus:
    """Determine overall health from component statuses."""
    if any(c.status == HealthStatus.UNHEALTHY for c in components):
        return HealthStatus.UNHEALTHY
    if any(c.status == HealthStatus.DEGRADED for c in components):
        return HealthStatus.DEGRADED
    return HealthStatus.HEALTHY


def run_health_checks(app_state: dict | None = None) -> HealthResponse:
    """Run all health checks and return aggregated response."""
    components = [
        check_database(),
        check_vectorstore(),
    ]
    if app_state is not None:
        components.append(check_models_loaded(app_state))

    overall = get_overall_status(components)
    return HealthResponse(status=overall, components=components)