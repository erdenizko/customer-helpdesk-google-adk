"""Integration tests for validation layer with session state.

These tests verify that validation functions work correctly when integrated
with session state. Tests cover valid flows, JSON string parsing, and
session state integration.
"""

import json

import pytest
from pydantic import ValidationError

from src.customer_helpdesk.services.validation import (
    ValidationParseError,
    get_with_validation,
    validate_classifier_output,
    validate_final_response,
    validate_history_context,
)


class TestValidationFlowWithSessionState:
    """Tests for validation functions working with session state simulation."""

    def test_valid_classifier_output_flows_through_session_state(self):
        session_state = {"classifier_intent": {"intent": "billing"}}
        result = get_with_validation(
            session_state, "classifier_intent", validate_classifier_output
        )
        assert result.intent == "billing"

    def test_valid_classifier_output_technical_intent(self):
        session_state = {"classifier_intent": {"intent": "technical"}}
        result = get_with_validation(
            session_state, "classifier_intent", validate_classifier_output
        )
        assert result.intent == "technical"

    def test_valid_classifier_output_general_intent(self):
        session_state = {"classifier_intent": {"intent": "general"}}
        result = get_with_validation(
            session_state, "classifier_intent", validate_classifier_output
        )
        assert result.intent == "general"

    def test_valid_history_context_flows_through_session_state(self):
        session_state = {
            "history_context": {
                "tickets": [
                    {
                        "id": "ticket-1",
                        "category": "technical",
                        "subject": "Login issue",
                        "status": "resolved",
                        "created_at": "2024-01-15T10:00:00",
                    }
                ],
                "similar_tickets": [
                    {
                        "id": "ticket-2",
                        "subject": "Password reset",
                        "resolution": "Reset via email",
                        "category": "technical",
                    }
                ],
            }
        }
        result = get_with_validation(
            session_state, "history_context", validate_history_context
        )
        assert len(result.tickets) == 1
        assert len(result.similar_tickets) == 1
        assert result.tickets[0].id == "ticket-1"
        assert result.similar_tickets[0].resolution == "Reset via email"

    def test_valid_history_context_empty_tickets(self):
        session_state = {
            "history_context": {
                "tickets": [],
                "similar_tickets": [
                    {
                        "id": "t-1",
                        "subject": "Issue",
                        "resolution": "Fixed",
                        "category": "billing",
                    }
                ],
            }
        }
        result = get_with_validation(
            session_state, "history_context", validate_history_context
        )
        assert result.tickets == []
        assert len(result.similar_tickets) == 1

    def test_valid_final_response_flows_through_session_state(self):
        session_state = {
            "final_response": {"response": "Your issue has been resolved."}
        }
        result = get_with_validation(
            session_state, "final_response", validate_final_response
        )
        assert result.response == "Your issue has been resolved."

    def test_valid_final_response_empty_string(self):
        session_state = {"final_response": {"response": ""}}
        result = get_with_validation(
            session_state, "final_response", validate_final_response
        )
        assert result.response == ""

    def test_json_string_stored_in_session_state_is_parsed(self):
        session_state = {"classifier_intent": '{"intent": "technical"}'}
        result = get_with_validation(
            session_state, "classifier_intent", validate_classifier_output
        )
        assert result.intent == "technical"

    def test_json_string_history_context_parsed(self):
        session_state = {"history_context": '{"tickets": [], "similar_tickets": []}'}
        result = get_with_validation(
            session_state, "history_context", validate_history_context
        )
        assert result.tickets == []
        assert result.similar_tickets == []

    def test_json_string_final_response_parsed(self):
        session_state = {"final_response": '{"response": "Help is on the way!"}'}
        result = get_with_validation(
            session_state, "final_response", validate_final_response
        )
        assert result.response == "Help is on the way!"

    def test_missing_session_state_key_raises_key_error(self):
        session_state = {}
        with pytest.raises(KeyError):
            get_with_validation(
                session_state, "missing_key", validate_classifier_output
            )

    def test_multiple_session_values_validated_sequentially(self):
        session_state = {
            "classifier_intent": {"intent": "billing"},
            "history_context": {
                "tickets": [
                    {
                        "id": "t-5",
                        "category": "billing",
                        "subject": "Invoice question",
                        "status": "open",
                        "created_at": "2024-02-10T14:30:00",
                    }
                ],
                "similar_tickets": [],
            },
            "final_response": {"response": "About your invoice..."},
        }

        intent = get_with_validation(
            session_state, "classifier_intent", validate_classifier_output
        )
        history = get_with_validation(
            session_state, "history_context", validate_history_context
        )
        response = get_with_validation(
            session_state, "final_response", validate_final_response
        )

        assert intent.intent == "billing"
        assert len(history.tickets) == 1
        assert "invoice" in response.response.lower()


class TestValidationInputTypes:
    """Tests for different input types to validation functions."""

    def test_validate_classifier_output_with_dict(self):
        result = validate_classifier_output({"intent": "billing"})
        assert result.intent == "billing"

    def test_validate_classifier_output_with_valid_json_string(self):
        result = validate_classifier_output('{"intent": "technical"}')
        assert result.intent == "technical"

    def test_validate_history_context_with_dict(self):
        data = {"tickets": [], "similar_tickets": []}
        result = validate_history_context(data)
        assert result.tickets == []

    def test_validate_history_context_with_json_string(self):
        result = validate_history_context('{"tickets": [], "similar_tickets": []}')
        assert result.tickets == []

    def test_validate_final_response_with_dict(self):
        result = validate_final_response({"response": "Test"})
        assert result.response == "Test"

    def test_validate_final_response_with_json_string(self):
        result = validate_final_response('{"response": "Test"}')
        assert result.response == "Test"


class TestEndToEndValidationFlow:
    """End-to-end tests for validation flow through the system."""

    def test_complete_valid_flow_all_validators(self):
        session_state = {
            "classifier_intent": {"intent": "technical"},
            "history_context": {
                "tickets": [
                    {
                        "id": "t-100",
                        "category": "technical",
                        "subject": "API error",
                        "status": "open",
                        "created_at": "2024-02-01T09:00:00",
                    }
                ],
                "similar_tickets": [
                    {
                        "id": "t-101",
                        "subject": "API timeout",
                        "resolution": "Increased timeout",
                        "category": "technical",
                    }
                ],
            },
            "final_response": {"response": "I've looked into your API issue."},
        }

        intent = get_with_validation(
            session_state, "classifier_intent", validate_classifier_output
        )
        history = get_with_validation(
            session_state, "history_context", validate_history_context
        )
        response = get_with_validation(
            session_state, "final_response", validate_final_response
        )

        assert intent.intent == "technical"
        assert len(history.tickets) == 1
        assert len(history.similar_tickets) == 1
        assert "API" in response.response

    def test_complete_flow_with_json_strings(self):
        session_state = {
            "classifier_intent": '{"intent": "billing"}',
            "history_context": '{"tickets": [], "similar_tickets": []}',
            "final_response": '{"response": "Billing inquiry processed."}',
        }

        intent = get_with_validation(
            session_state, "classifier_intent", validate_classifier_output
        )
        history = get_with_validation(
            session_state, "history_context", validate_history_context
        )
        response = get_with_validation(
            session_state, "final_response", validate_final_response
        )

        assert intent.intent == "billing"
        assert history.tickets == []
        assert response.response == "Billing inquiry processed."

    def test_mixed_dict_and_json_string_inputs(self):
        session_state = {
            "classifier_intent": {"intent": "general"},
            "history_context": '{"tickets": [], "similar_tickets": []}',
            "final_response": {"response": "General inquiry handled."},
        }

        intent = get_with_validation(
            session_state, "classifier_intent", validate_classifier_output
        )
        history = get_with_validation(
            session_state, "history_context", validate_history_context
        )
        response = get_with_validation(
            session_state, "final_response", validate_final_response
        )

        assert intent.intent == "general"
        assert history.tickets == []
        assert (
            "general" in response.response.lower()
            or "handled" in response.response.lower()
        )

    def test_complex_history_context_with_multiple_tickets(self):
        session_state = {
            "history_context": {
                "tickets": [
                    {
                        "id": "t-1",
                        "category": "billing",
                        "subject": "Invoice issue",
                        "status": "resolved",
                        "created_at": "2024-01-05T10:00:00",
                    },
                    {
                        "id": "t-2",
                        "category": "technical",
                        "subject": "Login problem",
                        "status": "open",
                        "created_at": "2024-02-20T15:30:00",
                    },
                ],
                "similar_tickets": [
                    {
                        "id": "t-3",
                        "subject": "Payment failed",
                        "resolution": "Updated payment method",
                        "category": "billing",
                    }
                ],
            }
        }

        result = get_with_validation(
            session_state, "history_context", validate_history_context
        )

        assert len(result.tickets) == 2
        assert len(result.similar_tickets) == 1
        assert result.tickets[0].id == "t-1"
        assert result.tickets[1].id == "t-2"
        assert result.similar_tickets[0].resolution == "Updated payment method"


class TestValidationWithMockedSessionState:
    """Tests using mocked session state for edge cases."""

    def test_session_state_with_none_value(self):
        session_state = {"classifier_intent": None}
        with pytest.raises((ValidationParseError, TypeError)):
            get_with_validation(
                session_state, "classifier_intent", validate_classifier_output
            )

    def test_session_state_with_boolean_true(self):
        session_state = {"classifier_intent": True}
        with pytest.raises((ValidationParseError, TypeError)):
            get_with_validation(
                session_state, "classifier_intent", validate_classifier_output
            )

    def test_session_state_with_boolean_false(self):
        session_state = {"classifier_intent": False}
        with pytest.raises((ValidationParseError, TypeError)):
            get_with_validation(
                session_state, "classifier_intent", validate_classifier_output
            )
