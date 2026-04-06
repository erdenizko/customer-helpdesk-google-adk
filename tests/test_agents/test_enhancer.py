import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.customer_helpdesk.agents.response_enhancer import agent


def test_enhancer_agent_definition():
    """Test that enhancer agent has correct definition."""
    assert agent.root_agent.name == "ResponseEnhancer"
    assert agent.root_agent.output_key == "final_response"


def test_rag_retrieve_function_exists():
    """Test that rag_retrieve tool is defined."""
    assert len(agent.RAG_TOOLS) == 1
    assert agent.RAG_TOOLS[0].name == "rag_retrieve"


@pytest.mark.asyncio
async def test_rag_skipped_for_non_technical():
    """Test that RAG is skipped for non-technical queries."""
    result = await agent.rag_retrieve("test query", "billing")
    assert result["skipped"] == True
    assert result["results"] == []


@pytest.mark.asyncio
async def test_rag_skipped_for_general():
    """Test that RAG is skipped for general queries."""
    result = await agent.rag_retrieve("test query", "general")
    assert result["skipped"] == True
    assert result["results"] == []


@pytest.mark.asyncio
async def test_rag_performed_for_technical():
    """Test that RAG is performed for technical queries."""
    mock_result = MagicMock()
    with patch(
        "src.customer_helpdesk.agents.response_enhancer.agent.get_vector_store",
        new_callable=AsyncMock,
    ) as mock_vs:
        mock_vs.return_value.search = AsyncMock(return_value=[mock_result])
        result = await agent.rag_retrieve("API error", "technical")
        assert result["skipped"] == False
        assert result["status"] == "success"
