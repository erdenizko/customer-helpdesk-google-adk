"""Safe JSON parsing utilities with fallback handling.

Provides fail-safe JSON parsing and serialization functions that log warnings
instead of raising exceptions when parsing fails.
"""

import json
from typing import Any

import structlog

logger = structlog.get_logger()

_MISSING = object()


def safe_json_parse(json_str: str | dict, fallback: dict = _MISSING) -> dict:
    """Parse JSON safely, returning fallback on failure.

    Args:
        json_str: A JSON string or already-parsed dict.
        fallback: Value to return on parse failure. Defaults to {}.

    Returns:
        Parsed dict on success, fallback on failure.
    """
    if fallback is _MISSING:
        fallback = {}

    if isinstance(json_str, dict):
        return json_str

    if json_str is None or json_str == "":
        logger.warning(
            "json_parse_failed",
            reason="empty_input",
            input_type=type(json_str).__name__,
        )
        return fallback

    # Try to parse string
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(
            "json_parse_failed",
            reason="JSONDecodeError",
            error=str(e),
            input=json_str[:100],
        )
        return fallback
    except TypeError as e:
        logger.warning(
            "json_parse_failed",
            reason="TypeError",
            error=str(e),
            input_type=type(json_str).__name__,
        )
        return fallback


def safe_json_dumps(obj: Any, fallback: str = "{}") -> str:
    """Serialize object to JSON safely, returning fallback on failure.

    Args:
        obj: Object to serialize.
        fallback: Value to return on serialization failure. Defaults to "{}".

    Returns:
        JSON string on success, fallback on failure.
    """
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError) as e:
        logger.warning(
            "json_dumps_failed",
            reason=type(e).__name__,
            error=str(e),
            obj_type=type(obj).__name__,
        )
        return fallback
