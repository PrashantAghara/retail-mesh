"""Unified application exceptions.

All domain and infrastructure code should raise these exceptions so API
layers can translate them into consistent HTTP responses.
"""


class AppError(Exception):
    """Base class for all application-level errors."""

    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ModelLoadError(AppError):
    """Raised when ML models fail to load or initialize."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class ConfigurationError(AppError):
    """Raised when application configuration is invalid or missing."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404)


class ValidationError(AppError):
    """Raised when input validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)