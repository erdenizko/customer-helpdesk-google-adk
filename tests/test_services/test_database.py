import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_get_user_tickets_returns_list():
    """Test that get_user_tickets returns a list."""
    with patch(
        "src.customer_helpdesk.services.database.async_session_factory"
    ) as mock_factory:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        mock_factory.return_value.__aenter__.return_value = mock_session

        from src.customer_helpdesk.services.database import get_user_tickets

        result = await get_user_tickets("user123")
        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_search_similar_tickets_returns_list():
    """Test that search_similar_tickets returns a list."""
    with patch(
        "src.customer_helpdesk.services.database.async_session_factory"
    ) as mock_factory:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        mock_factory.return_value.__aenter__.return_value = mock_session

        from src.customer_helpdesk.services.database import search_similar_tickets

        result = await search_similar_tickets("test query", "technical")
        assert isinstance(result, list)
