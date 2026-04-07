"""Validation layer for agent outputs.

This module provides validation functions for agent outputs stored in session state.
It ensures that data retrieved from session.state is properly validated before use.
"""

import json
import logging
from typing import Callable, TypeVar

from pydantic import ValidationError

from src.customer_helpdesk.models.agent_outputs import (
    ClassifierOutput,
    FinalResponse,
    HistoryContext,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ValidationParseError(Exception):
    """Raised when JSON parsing fails during validation."""

    pass


def _parse_json_string(data: str) -> dict:
    """Parse a JSON string into a dict, raising ValidationParseError on failure."""
    if not data or not data.strip():
        logger.warning(
            "validation_parse_error",
            error="Input cannot be empty or whitespace-only",
            data_type="str",
        )
        raise ValidationParseError("Input cannot be empty or whitespace-only")
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        logger.warning(
            "validation_parse_error", error=f"Invalid JSON: {e}", data_type="str"
        )
        raise ValidationParseError(f"Invalid JSON: {e}") from e


def validate_classifier_output(data: str | dict) -> ClassifierOutput:
    """Validate classifier output data."""
    if isinstance(data, dict):
        parsed = data
    elif isinstance(data, str):
        parsed = _parse_json_string(data)
    else:
        logger.warning(
            "validation_parse_error",
            error=f"Expected str or dict, got {type(data).__name__}",
            data_type=type(data).__name__,
        )
        raise ValidationParseError(f"Expected str or dict, got {type(data).__name__}")

    try:
        return ClassifierOutput.model_validate(parsed)
    except ValidationError as e:
        logger.warning("validation_error", error=str(e), data_type="ClassifierOutput")
        raise


def validate_history_context(data: str | dict) -> HistoryContext:
    """Validate history context data."""
    if isinstance(data, dict):
        parsed = data
    elif isinstance(data, str):
        parsed = _parse_json_string(data)
    else:
        logger.warning(
            "validation_parse_error",
            error=f"Expected str or dict, got {type(data).__name__}",
            data_type=type(data).__name__,
        )
        raise ValidationParseError(f"Expected str or dict, got {type(data).__name__}")

    try:
        return HistoryContext.model_validate(parsed)
    except ValidationError as e:
        logger.warning("validation_error", error=str(e), data_type="HistoryContext")
        raise


def validate_final_response(data: str | dict) -> FinalResponse:
    """Validate final response data."""
    if isinstance(data, dict):
        parsed = data
    elif isinstance(data, str):
        parsed = _parse_json_string(data)
    else:
        logger.warning(
            "validation_parse_error",
            error=f"Expected str or dict, got {type(data).__name__}",
            data_type=type(data).__name__,
        )
        raise ValidationParseError(f"Expected str or dict, got {type(data).__name__}")

    try:
        return FinalResponse.model_validate(parsed)
    except ValidationError as e:
        logger.warning("validation_error", error=str(e), data_type="FinalResponse")
        raise


def get_with_validation(
    session_state: dict, key: str, validator: Callable[[str | dict], T]
) -> T:
    """Get a value from session state and validate it."""
    value = session_state[key]
    return validator(value)
