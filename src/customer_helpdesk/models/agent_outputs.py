"""Pydantic models for agent outputs.

These models validate the structured outputs from the ADK agents:
- ClassifierOutput: intent classification (billing/technical/general)
- TicketInfo: user's ticket data from history
- SimilarTicket: similar resolved issues for RAG context
- HistoryContext: combined user tickets and similar tickets
- FinalResponse: the enhanced response text
"""

from typing import Literal

from pydantic import BaseModel


class ClassifierOutput(BaseModel):
    """Output from the Classifier agent.

    Classifies user query into one of three intent categories.
    """

    intent: Literal["billing", "technical", "general"]

    model_config = {"extra": "forbid"}


class TicketInfo(BaseModel):
    """Information about a user's ticket from history."""

    id: str
    category: str
    subject: str
    status: str
    created_at: str

    model_config = {"extra": "forbid"}


class SimilarTicket(BaseModel):
    """A similar resolved ticket for RAG context."""

    id: str
    subject: str
    resolution: str | None
    category: str

    model_config = {"extra": "forbid"}


class HistoryContext(BaseModel):
    """Combined history context from HistoryCheck agent.

    Contains user's ticket history and similar resolved issues.
    """

    tickets: list[TicketInfo]
    similar_tickets: list[SimilarTicket]

    model_config = {"extra": "forbid"}


class FinalResponse(BaseModel):
    """Final response from the ResponseEnhancer agent."""

    response: str

    model_config = {"extra": "forbid"}
