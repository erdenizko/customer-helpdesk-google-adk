"""Tests for agent output models."""

import pytest
from pydantic import ValidationError

from src.customer_helpdesk.models.agent_outputs import (
    ClassifierOutput,
    TicketInfo,
    SimilarTicket,
    HistoryContext,
    FinalResponse,
)


class TestClassifierOutput:
    """Tests for ClassifierOutput model."""

    def test_valid_billing_intent(self):
        """Test valid billing intent parsing."""
        result = ClassifierOutput.model_validate({"intent": "billing"})
        assert result.intent == "billing"

    def test_valid_technical_intent(self):
        """Test valid technical intent parsing."""
        result = ClassifierOutput.model_validate({"intent": "technical"})
        assert result.intent == "technical"

    def test_valid_general_intent(self):
        """Test valid general intent parsing."""
        result = ClassifierOutput.model_validate({"intent": "general"})
        assert result.intent == "general"

    def test_invalid_intent_rejected(self):
        """Test that invalid intent value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ClassifierOutput.model_validate({"intent": "unknown"})
        assert "intent" in str(exc_info.value)

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ClassifierOutput.model_validate({"intent": "billing", "extra": "value"})
        assert "extra" in str(exc_info.value)


class TestTicketInfo:
    """Tests for TicketInfo model."""

    def test_valid_ticket_info(self):
        """Test valid TicketInfo parsing."""
        data = {
            "id": "ticket-123",
            "category": "billing",
            "subject": "Help with invoice",
            "status": "open",
            "created_at": "2024-01-15T10:30:00",
        }
        result = TicketInfo.model_validate(data)
        assert result.id == "ticket-123"
        assert result.category == "billing"

    def test_ticket_info_missing_field(self):
        """Test that missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            TicketInfo.model_validate({"id": "ticket-123"})


class TestSimilarTicket:
    """Tests for SimilarTicket model."""

    def test_valid_similar_ticket(self):
        """Test valid SimilarTicket parsing."""
        data = {
            "id": "ticket-456",
            "subject": "API error fix",
            "resolution": "Restart the service",
            "category": "technical",
        }
        result = SimilarTicket.model_validate(data)
        assert result.id == "ticket-456"
        assert result.resolution == "Restart the service"

    def test_similar_ticket_null_resolution(self):
        """Test SimilarTicket with null resolution."""
        data = {
            "id": "ticket-789",
            "subject": "Unknown issue",
            "resolution": None,
            "category": "general",
        }
        result = SimilarTicket.model_validate(data)
        assert result.resolution is None


class TestHistoryContext:
    """Tests for HistoryContext model."""

    def test_valid_history_context(self):
        """Test valid HistoryContext parsing."""
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
        result = HistoryContext.model_validate(data)
        assert len(result.tickets) == 1
        assert len(result.similar_tickets) == 1

    def test_history_context_empty_lists(self):
        """Test HistoryContext with empty lists."""
        data = {"tickets": [], "similar_tickets": []}
        result = HistoryContext.model_validate(data)
        assert result.tickets == []
        assert result.similar_tickets == []


class TestFinalResponse:
    """Tests for FinalResponse model."""

    def test_valid_final_response(self):
        """Test valid FinalResponse parsing."""
        data = {"response": "Here is your helpful answer."}
        result = FinalResponse.model_validate(data)
        assert result.response == "Here is your helpful answer."

    def test_empty_response(self):
        """Test FinalResponse with empty string is valid."""
        data = {"response": ""}
        result = FinalResponse.model_validate(data)
        assert result.response == ""

    def test_missing_response_field(self):
        """Test that missing response field raises ValidationError."""
        with pytest.raises(ValidationError):
            FinalResponse.model_validate({})

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            FinalResponse.model_validate({"response": "test", "extra": "value"})
