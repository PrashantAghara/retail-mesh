"""Unified application exceptions.

All domain and infrastructure code should raise these exceptions so API
layers can translate them into consistent HTTP responses.
"""

from typing import Any


class AppError(Exception):
    """Base class for all application-level errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}


class ModelLoadError(AppError):
    """Raised when ML models fail to load or initialize."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=500, error_code="MODEL_LOAD_ERROR", details=details)


class ConfigurationError(AppError):
    """Raised when application configuration is invalid or missing."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=500, error_code="CONFIGURATION_ERROR", details=details)


class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=404, error_code="NOT_FOUND", details=details)


class ValidationError(AppError):
    """Raised when input validation fails."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=422, error_code="VALIDATION_ERROR", details=details)


class ExternalServiceError(AppError):
    """Raised when an external service (API, database) fails."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=502, error_code="EXTERNAL_SERVICE_ERROR", details=details)