"""Tool input validation helpers for customer helpdesk agents."""

import functools
import inspect
import re
from enum import Enum
from typing import Any, Callable, get_type_hints


class ValidationErrorCode(Enum):
    """Error codes for ValidationError."""

    EMPTY_INPUT = "EMPTY_INPUT"
    INVALID_TYPE = "INVALID_TYPE"
    INVALID_VALUE = "INVALID_VALUE"
    COERCION_FAILED = "COERCION_FAILED"


class ValidationError(Exception):
    """Exception raised when tool input validation fails.

    Attributes:
        code: Error code from ValidationErrorCode enum
        message: Human-readable error message
        field: Optional field name that failed validation
    """

    def __init__(
        self,
        code: ValidationErrorCode,
        message: str,
        field: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field

    def __repr__(self) -> str:
        parts = [f"ValidationError(code={self.code.value}"]
        if self.field:
            parts.append(f", field={self.field}")
        parts.append(f", message={self.message!r})")
        return "".join(parts)


# Maximum string length after sanitization
MAX_STRING_LENGTH = 10000

# SQL injection patterns (basic detection)
SQL_INJECTION_PATTERNS = [
    r"(\bOR\b|\bAND\b).*=.*",  # OR 1=1 OR AND id=
    r"(--|;#|/\*|\*/)",  # SQL comments
    r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)",  # SQL keywords
    r"(';|--)",  # Quote with comment
]


def sanitize_string(input: str) -> str:
    """Sanitize a string input to prevent injection attacks.

    Performs the following:
    - Strips leading/trailing whitespace
    - Removes potential SQL injection patterns (basic)
    - Limits length to MAX_STRING_LENGTH (10000 chars)

    Args:
        input: The string to sanitize

    Returns:
        Sanitized string safe for use
    """
    if not isinstance(input, str):
        return input

    # Strip whitespace
    result = input.strip()

    # Remove SQL injection patterns
    for pattern in SQL_INJECTION_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    # Limit length
    if len(result) > MAX_STRING_LENGTH:
        result = result[:MAX_STRING_LENGTH]

    return result


def _coerce_type(value: Any, target_type: type) -> tuple[bool, Any]:
    """Attempt to coerce a value to a target type.

    Args:
        value: The value to coerce
        target_type: The target type

    Returns:
        Tuple of (success, coerced_value or original_value)
    """
    if isinstance(value, target_type):
        return True, value

    # Handle str to int coercion
    if target_type is int and isinstance(value, str):
        try:
            return True, int(value)
        except ValueError:
            return False, value

    # Handle str to float coercion
    if target_type is float and isinstance(value, str):
        try:
            return True, float(value)
        except ValueError:
            return False, value

    # Handle str to bool coercion
    if target_type is bool and isinstance(value, str):
        lower_val = value.lower()
        if lower_val in ("true", "1", "yes"):
            return True, True
        if lower_val in ("false", "0", "no"):
            return True, False
        return False, value

    return False, value


def validate_tool_input(func: Callable) -> Callable:
    """Decorator to validate and coerce tool function inputs.

    Performs the following validations:
    - Required parameters (no default) must not be empty strings
    - Type coercion for common cases (str to int, str to float)
    - Calls original function with validated/coerced arguments

    Args:
        func: The tool function to wrap with validation

    Returns:
        Wrapped function with validation
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        return await _validate_and_call(func, args, kwargs)

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        return _validate_and_call(func, args, kwargs)

    # Determine if function is async
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def _validate_and_call(func: Callable, args: tuple, kwargs: dict) -> Any:
    """Validate arguments and call the function.

    Args:
        func: The function to call
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Result from the function call

    Raises:
        ValidationError: If validation fails
    """
    sig = inspect.signature(func)
    param_map = dict(zip(sig.parameters.keys(), args))

    # Merge kwargs into param_map
    param_map.update(kwargs)

    # Get type hints
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}

    validated_params = {}

    for param_name, param in sig.parameters.items():
        value = param_map.get(param_name)
        target_type = type_hints.get(param_name)

        # Check if param is required (no default)
        is_required = param.default is inspect.Parameter.empty

        # Skip variadic parameters
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            validated_params[param_name] = value
            continue

        # Handle missing required params
        if value is None:
            if is_required:
                raise ValidationError(
                    code=ValidationErrorCode.EMPTY_INPUT,
                    message=f"Required parameter '{param_name}' is missing",
                    field=param_name,
                )
            validated_params[param_name] = value
            continue

        # Sanitize string inputs
        if isinstance(value, str):
            value = sanitize_string(value)
            # Check for empty after sanitization
            if not value and is_required:
                raise ValidationError(
                    code=ValidationErrorCode.EMPTY_INPUT,
                    message=f"Required parameter '{param_name}' cannot be empty",
                    field=param_name,
                )

        # Type coercion for string inputs
        if (
            target_type
            and isinstance(value, str)
            and not isinstance(value, target_type)
        ):
            success, coerced = _coerce_type(value, target_type)
            if success:
                value = coerced
            else:
                # If coercion failed and it's a required type mismatch
                if not isinstance(value, target_type):
                    raise ValidationError(
                        code=ValidationErrorCode.COERCION_FAILED,
                        message=f"Cannot coerce '{param_name}' to {target_type.__name__}",
                        field=param_name,
                    )

        validated_params[param_name] = value

    return func(**validated_params)
