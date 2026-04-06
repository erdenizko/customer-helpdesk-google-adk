import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_vector_store_search_returns_list():
    """Test that VectorStoreService.search returns a list."""
    with patch(
        "src.customer_helpdesk.services.vector_store.AsyncQdrantClient"
    ) as mock_client:
        mock_instance = AsyncMock()
        mock_instance.search.return_value = []
        mock_client.return_value = mock_instance

        from src.customer_helpdesk.services.vector_store import VectorStoreService

        vs = VectorStoreService()
        result = await vs.search([0.1] * 1536)
        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_vector_store_upsert():
    """Test that VectorStoreService.upsert works."""
    with patch(
        "src.customer_helpdesk.services.vector_store.AsyncQdrantClient"
    ) as mock_client:
        mock_instance = AsyncMock()
        mock_instance.upsert.return_value = None
        mock_client.return_value = mock_instance

        from src.customer_helpdesk.services.vector_store import VectorStoreService

        vs = VectorStoreService()
        # Should not raise
        await vs.upsert([{"id": "1", "vector": [0.1] * 1536, "payload": {}}])
        mock_instance.upsert.assert_called_once()
