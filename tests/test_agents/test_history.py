import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.customer_helpdesk.agents.history_check import agent


def test_history_agent_definition():
    """Test that history agent has correct definition."""
    assert agent.root_agent.name == "HistoryCheck"
    assert agent.root_agent.output_key == "history_context"


def test_history_tools_exist():
    """Test that history tools are defined."""
    assert len(agent.HISTORY_TOOLS) == 2
    tool_names = [t.name for t in agent.HISTORY_TOOLS]
    assert "lookup_user_history" in tool_names
    assert "lookup_similar_issues" in tool_names


@pytest.mark.asyncio
async def test_lookup_user_history_returns_dict():
    """Test that lookup_user_history returns expected structure."""
    mock_ticket = MagicMock()
    mock_ticket.id = "ticket-1"
    mock_ticket.category.value = "technical"
    mock_ticket.subject = "API Error"
    mock_ticket.status.value = "resolved"
    mock_ticket.created_at.isoformat.return_value = "2024-01-01T00:00:00"

    with patch(
        "src.customer_helpdesk.agents.history_check.agent.get_user_tickets",
        new_callable=AsyncMock,
    ) as mock:
        mock.return_value = [mock_ticket]
        result = await agent.lookup_user_history("user123")
        assert result["status"] == "success"
        assert "tickets" in result
        assert len(result["tickets"]) == 1
        assert result["tickets"][0]["id"] == "ticket-1"


@pytest.mark.asyncio
async def test_lookup_similar_issues_returns_dict():
    """Test that lookup_similar_issues returns expected structure."""
    mock_ticket = MagicMock()
    mock_ticket.id = "ticket-2"
    mock_ticket.subject = "500 Error"
    mock_ticket.resolution = "Fixed in v2.0"
    mock_ticket.category.value = "technical"

    with patch(
        "src.customer_helpdesk.agents.history_check.agent.search_similar_tickets",
        new_callable=AsyncMock,
    ) as mock:
        mock.return_value = [mock_ticket]
        result = await agent.lookup_similar_issues("test query", "technical")
        assert result["status"] == "success"
        assert "similar_tickets" in result
        assert len(result["similar_tickets"]) == 1
        assert result["similar_tickets"][0]["id"] == "ticket-2"
