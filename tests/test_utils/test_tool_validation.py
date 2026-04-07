"""Tests for tool validation utilities."""

import pytest

from src.customer_helpdesk.utils.tool_validation import (
    ValidationError,
    ValidationErrorCode,
    sanitize_string,
    validate_tool_input,
)


class TestSanitizeString:
    """Tests for sanitize_string function."""

    def test_strips_whitespace(self):
        result = sanitize_string("  hello world  ")
        assert result == "hello world"

    def test_removes_sql_injection_or_pattern(self):
        result = sanitize_string("test' OR '1'='1")
        assert "' OR" not in result
        assert "OR" not in result

    def test_removes_sql_injection_union_select(self):
        result = sanitize_string("'; DROP TABLE users; --")
        assert "DROP" not in result
        assert "--" not in result

    def test_limits_length(self):
        long_string = "a" * 15000
        result = sanitize_string(long_string)
        assert len(result) == 10000

    def test_preserves_normal_string(self):
        result = sanitize_string("Hello, World!")
        assert result == "Hello, World!"

    def test_non_string_passthrough(self):
        result = sanitize_string(123)
        assert result == 123


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_error_code_attribute(self):
        error = ValidationError(
            code=ValidationErrorCode.EMPTY_INPUT,
            message="Field is required",
            field="user_id",
        )
        assert error.code == ValidationErrorCode.EMPTY_INPUT
        assert error.message == "Field is required"
        assert error.field == "user_id"

    def test_error_repr(self):
        error = ValidationError(
            code=ValidationErrorCode.INVALID_TYPE,
            message="Type mismatch",
            field="limit",
        )
        assert "INVALID_TYPE" in repr(error)
        assert "limit" in repr(error)


class TestValidateToolInput:
    """Tests for validate_tool_input decorator."""

    def test_valid_input_passes_through(self):
        @validate_tool_input
        def simple_tool(user_id: str, limit: int = 5) -> dict:
            return {"user_id": user_id, "limit": limit}

        result = simple_tool("user123", limit=10)
        assert result == {"user_id": "user123", "limit": 10}

    def test_empty_string_raises_validation_error(self):
        @validate_tool_input
        def lookup_tool(user_id: str) -> dict:
            return {"user_id": user_id}

        with pytest.raises(ValidationError) as exc_info:
            lookup_tool("")
        assert exc_info.value.code == ValidationErrorCode.EMPTY_INPUT
        assert exc_info.value.field == "user_id"

    def test_string_to_int_coercion(self):
        @validate_tool_input
        def lookup_tool(user_id: str, limit: int = 5) -> dict:
            return {"user_id": user_id, "limit": limit}

        result = lookup_tool("user123", limit="10")
        assert result["limit"] == 10
        assert isinstance(result["limit"], int)

    def test_string_to_float_coercion(self):
        @validate_tool_input
        def rate_tool(rate: float) -> dict:
            return {"rate": rate}

        result = rate_tool("3.14")
        assert result["rate"] == 3.14
        assert isinstance(result["rate"], float)

    def test_invalid_coercion_raises_error(self):
        @validate_tool_input
        def count_tool(count: int) -> dict:
            return {"count": count}

        with pytest.raises(ValidationError) as exc_info:
            count_tool("not_a_number")
        assert exc_info.value.code == ValidationErrorCode.COERCION_FAILED

    def test_missing_required_param_raises_error(self):
        @validate_tool_input
        def lookup_tool(user_id: str, category: str) -> dict:
            return {"user_id": user_id, "category": category}

        with pytest.raises(ValidationError) as exc_info:
            lookup_tool("user123")
        assert exc_info.value.code == ValidationErrorCode.EMPTY_INPUT
        assert exc_info.value.field == "category"

    def test_optional_param_can_be_none(self):
        @validate_tool_input
        def lookup_tool(user_id: str, limit: int | None = None) -> dict:
            return {"user_id": user_id, "limit": limit}

        result = lookup_tool("user123")
        assert result["limit"] is None

    def test_sanitization_applied_to_strings(self):
        @validate_tool_input
        def lookup_tool(query: str) -> dict:
            return {"query": query}

        result = lookup_tool("  test'; DROP TABLE--  ")
        assert "DROP" not in result["query"]
        assert "--" not in result["query"]
        assert "test" in result["query"]

    @pytest.mark.asyncio
    async def test_async_function_support(self):
        @validate_tool_input
        async def async_tool(user_id: str) -> dict:
            return {"user_id": user_id}

        result = await async_tool("user456")
        assert result["user_id"] == "user456"

    def test_bool_coercion_from_strings(self):
        @validate_tool_input
        def flag_tool(enabled: bool) -> dict:
            return {"enabled": enabled}

        assert flag_tool("true")["enabled"] is True
        assert flag_tool("false")["enabled"] is False
        assert flag_tool("yes")["enabled"] is True
        assert flag_tool("no")["enabled"] is False
        assert flag_tool("1")["enabled"] is True
        assert flag_tool("0")["enabled"] is False
