"""Tests for agent output validation functions."""

import pytest
from pydantic import ValidationError

from src.customer_helpdesk.services.validation import (
    ValidationParseError,
    get_with_validation,
    validate_classifier_output,
    validate_final_response,
    validate_history_context,
)


class TestValidateClassifierOutput:
    """Tests for validate_classifier_output function."""

    def test_valid_dict_input_passes_validation(self):
        """Test that valid dict input passes validation."""
        data = {"intent": "billing"}
        result = validate_classifier_output(data)
        assert result.intent == "billing"

    def test_valid_json_string_is_parsed_and_validated(self):
        """Test that valid JSON string is parsed and validated."""
        json_str = '{"intent": "technical"}'
        result = validate_classifier_output(json_str)
        assert result.intent == "technical"

    def test_valid_general_intent(self):
        """Test validation with general intent."""
        result = validate_classifier_output({"intent": "general"})
        assert result.intent == "general"

    def test_invalid_intent_raises_validation_error(self):
        """Test that invalid intent value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_classifier_output({"intent": "unknown"})
        assert "intent" in str(exc_info.value)

    def test_missing_intent_field_raises_validation_error(self):
        """Test that missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_classifier_output({})

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected (forbid mode)."""
        with pytest.raises(ValidationError) as exc_info:
            validate_classifier_output({"intent": "billing", "extra": "value"})
        assert "extra" not in str(exc_info.value) or "extra" in str(exc_info.value)

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValidationParseError."""
        with pytest.raises(ValidationParseError):
            validate_classifier_output("")

    def test_whitespace_string_raises_error(self):
        """Test that whitespace-only string raises ValidationParseError."""
        with pytest.raises(ValidationParseError):
            validate_classifier_output("   ")

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON string raises ValidationParseError."""
        with pytest.raises(ValidationParseError):
            validate_classifier_output("{invalid json}")

    def test_non_dict_input_raises_error(self):
        """Test that non-dict/non-string input raises ValidationParseError."""
        with pytest.raises(ValidationParseError):
            validate_classifier_output(123)


class TestValidateHistoryContext:
    """Tests for validate_history_context function."""

    def test_valid_dict_input_passes_validation(self):
        """Test that valid dict input passes validation."""
        data = {
            "tickets": [
                {
                    "id": "ticket-1",
                    "category": "billing",
                    "subject": "Invoice help",
                    "status": "resolved",
                    "created_at": "2024-01-10T08:00:00",
                }
            ],
            "similar_tickets": [
                {
                    "id": "ticket-2",
                    "subject": "Old invoice issue",
                    "resolution": "Updated billing info",
                    "category": "billing",
                }
            ],
        }
        result = validate_history_context(data)
        assert len(result.tickets) == 1
        assert len(result.similar_tickets) == 1

    def test_valid_json_string_is_parsed_and_validated(self):
        """Test that valid JSON string is parsed and validated."""
        json_str = '{"tickets": [], "similar_tickets": []}'
        result = validate_history_context(json_str)
        assert result.tickets == []
        assert result.similar_tickets == []

    def test_missing_tickets_field_raises_validation_error(self):
        """Test that missing tickets field raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_history_context({"similar_tickets": []})

    def test_missing_similar_tickets_field_raises_validation_error(self):
        """Test that missing similar_tickets field raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_history_context({"tickets": []})

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected (forbid mode)."""
        with pytest.raises(ValidationError):
            validate_history_context(
                {"tickets": [], "similar_tickets": [], "extra": "value"}
            )

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValidationParseError."""
        with pytest.raises(ValidationParseError):
            validate_history_context("")

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON string raises ValidationParseError."""
        with pytest.raises(ValidationParseError):
            validate_history_context("{not valid json}")


class TestValidateFinalResponse:
    """Tests for validate_final_response function."""

    def test_valid_dict_input_passes_validation(self):
        """Test that valid dict input passes validation."""
        data = {"response": "Here is your helpful answer."}
        result = validate_final_response(data)
        assert result.response == "Here is your helpful answer."

    def test_valid_json_string_is_parsed_and_validated(self):
        """Test that valid JSON string is parsed and validated."""
        json_str = '{"response": "Test response"}'
        result = validate_final_response(json_str)
        assert result.response == "Test response"

    def test_empty_response_string_is_valid(self):
        """Test that empty response string is valid."""
        result = validate_final_response({"response": ""})
        assert result.response == ""

    def test_missing_response_field_raises_validation_error(self):
        """Test that missing response field raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_final_response({})

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected (forbid mode)."""
        with pytest.raises(ValidationError):
            validate_final_response({"response": "test", "extra": "value"})

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValidationParseError."""
        with pytest.raises(ValidationParseError):
            validate_final_response("")

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON string raises ValidationParseError."""
        with pytest.raises(ValidationParseError):
            validate_final_response("not valid json")


class TestGetWithValidation:
    """Tests for get_with_validation helper function."""

    def test_get_valid_value_from_session_state(self):
        """Test getting and validating a valid value from session state."""
        session_state = {"classifier_intent": {"intent": "billing"}}
        result = get_with_validation(
            session_state, "classifier_intent", validate_classifier_output
        )
        assert result.intent == "billing"

    def test_get_value_raises_key_error_when_missing(self):
        """Test that KeyError is raised when key is not in session state."""
        session_state = {}
        with pytest.raises(KeyError):
            get_with_validation(
                session_state, "missing_key", validate_classifier_output
            )

    def test_get_invalid_value_raises_validation_error(self):
        """Test that ValidationError is raised when value is invalid."""
        session_state = {"classifier_intent": {"intent": "invalid"}}
        with pytest.raises(ValidationError):
            get_with_validation(
                session_state, "classifier_intent", validate_classifier_output
            )

    def test_get_json_string_value_from_session_state(self):
        """Test getting a JSON string value and validating it."""
        session_state = {"history_context": '{"tickets": [], "similar_tickets": []}'}
        result = get_with_validation(
            session_state, "history_context", validate_history_context
        )
        assert result.tickets == []
        assert result.similar_tickets == []

    def test_get_final_response_from_session_state(self):
        """Test getting and validating final response from session state."""
        session_state = {"final_response": {"response": "Test response text"}}
        result = get_with_validation(
            session_state, "final_response", validate_final_response
        )
        assert result.response == "Test response text"
