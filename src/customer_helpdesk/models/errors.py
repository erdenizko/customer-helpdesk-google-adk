"""Pydantic models for structured error responses.

These models ensure consistent error response format across the API:
- ErrorResponse: structured error with code, message, and correlation ID
- ErrorCode: enumerated error codes for categorization
"""

from enum import Enum

from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Error codes for categorization of errors.

    Each code represents a category of errors that can occur in the system.
    """

    AGENT_ERROR = "AGENT_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SESSION_ERROR = "SESSION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    """Structured error response for API errors.

    This model provides a consistent error format that:
    - Includes an error code for programmatic handling
    - Provides a user-friendly message (no internal details)
    - Includes a correlation ID for log tracing

    Internal details like stack traces are logged server-side
    but never exposed to clients.
    """

    error_code: str
    message: str
    correlation_id: str | None = None

    model_config = {"extra": "forbid"}
