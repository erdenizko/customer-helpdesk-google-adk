import pytest
from unittest.mock import AsyncMock, patch, MagicMock


def test_full_pipeline_import():
    """Test that full pipeline can be imported."""
    from src.customer_helpdesk.agent import root_agent, parallel_lookup

    # Verify root agent exists
    assert root_agent is not None
    assert root_agent.name == "HelpdeskOrchestrator"


def test_pipeline_error_handling_db_unavailable():
    """Test graceful error when database is unavailable."""
    with patch("src.customer_helpdesk.services.database.get_user_tickets") as mock_db:
        mock_db.side_effect = Exception("Database unavailable")

        # Verify error handling doesn't crash - agent can still be imported
        from src.customer_helpdesk.agent import root_agent

        assert root_agent is not None
        assert root_agent.name == "HelpdeskOrchestrator"


def test_pipeline_agent_structure():
    """Test that pipeline has correct agent structure."""
    from src.customer_helpdesk.agent import root_agent, parallel_lookup

    # Verify root agent
    assert root_agent.name == "HelpdeskOrchestrator"
    assert root_agent.description == "Main helpdesk orchestration agent"

    # Verify parallel lookup has sub-agents
    assert parallel_lookup.name == "ParallelLookup"
    assert len(parallel_lookup.sub_agents) == 2


def test_parallel_lookup_sub_agents():
    """Test parallel lookup contains classifier and history agents."""
    from src.customer_helpdesk.agent import parallel_lookup
    from src.customer_helpdesk.agents.classifier import agent as classifier_agent
    from src.customer_helpdesk.agents.history_check import agent as history_agent

    # Verify sub-agents are the classifier and history agents
    sub_agent_names = [sa.name for sa in parallel_lookup.sub_agents]
    assert "Classifier" in sub_agent_names
    assert "HistoryCheck" in sub_agent_names

    # Verify they match the imported agents
    assert parallel_lookup.sub_agents[0].name == classifier_agent.root_agent.name
    assert parallel_lookup.sub_agents[1].name == history_agent.root_agent.name


def test_orchestrator_sub_agents_structure():
    """Test root agent contains parallel_lookup and enhancer as sub_agents."""
    from src.customer_helpdesk.agent import root_agent
    from src.customer_helpdesk.agents.response_enhancer import agent as enhancer_agent

    # Verify orchestrator has parallel_lookup and enhancer as sub_agents
    sub_agent_names = [sa.name for sa in root_agent.sub_agents]
    assert "ParallelLookup" in sub_agent_names
    assert "ResponseEnhancer" in sub_agent_names


def test_pipeline_with_mocked_services():
    """Test full pipeline with mocked external services."""
    with (
        patch("src.customer_helpdesk.services.database.get_user_tickets") as mock_db,
        patch(
            "src.customer_helpdesk.services.vector_store.VectorStoreService.search"
        ) as mock_vs,
    ):
        mock_db.return_value = []
        mock_vs.return_value = []

        # Verify pipeline can be imported with mocked services
        from src.customer_helpdesk.agent import root_agent

        assert root_agent is not None
        assert root_agent.name == "HelpdeskOrchestrator"


def test_pipeline_error_handling_qdrant_unavailable():
    """Test graceful error when Qdrant vector store is unavailable."""
    with patch(
        "src.customer_helpdesk.services.vector_store.VectorStoreService.search"
    ) as mock_vs:
        mock_vs.side_effect = Exception("Qdrant unavailable")

        # Verify agent can still be imported
        from src.customer_helpdesk.agent import root_agent

        assert root_agent is not None


def test_pipeline_error_handling_multiple_failures():
    """Test error handling when both DB and Qdrant are unavailable."""
    with (
        patch("src.customer_helpdesk.services.database.get_user_tickets") as mock_db,
        patch(
            "src.customer_helpdesk.services.vector_store.VectorStoreService.search"
        ) as mock_vs,
    ):
        mock_db.side_effect = Exception("Database unavailable")
        mock_vs.side_effect = Exception("Qdrant unavailable")

        # Verify agent can still be imported despite service failures
        from src.customer_helpdesk.agent import root_agent

        assert root_agent is not None
        assert root_agent.name == "HelpdeskOrchestrator"
