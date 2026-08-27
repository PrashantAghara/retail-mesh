from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


_async_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker | None = None


def get_async_engine() -> AsyncEngine:
    """Get or create the async SQLAlchemy engine (singleton)."""
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        # Convert sync URL to async (postgresql+psycopg2:// -> postgresql+asyncpg://)
        async_url = settings.database_url.replace(
            "postgresql+psycopg2://", "postgresql+asyncpg://"
        ).replace("postgresql://", "postgresql+asyncpg://")
        _async_engine = create_async_engine(
            async_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _async_engine


def get_async_session_factory() -> async_sessionmaker:
    """Get or create the async session factory (singleton)."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _async_session_factory


@asynccontextmanager
async def async_session_scope() -> AsyncSession:
    """Async context manager for database sessions with automatic commit/rollback/close."""
    session = get_async_session_factory()()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_async_db_session() -> AsyncSession:
    """FastAPI dependency for async database sessions."""
    session = get_async_session_factory()()
    try:
        yield session
    finally:
        await session.close()


AsyncDbSessionDep = get_async_db_session


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass