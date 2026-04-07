"""Integration tests for session service with validation."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings before importing session_service."""
    mock_cfg = MagicMock()
    mock_cfg.database_url = "postgresql+asyncpg://test:test@localhost/test"
    mock_cfg.qdrant_url = "http://localhost:6333"
    mock_cfg.qdrant_collection = "test_collection"
    mock_cfg.openai_api_key = "sk-test"
    mock_cfg.basic_model = "openai/gpt-4o-mini"
    mock_cfg.complex_model = "openai/gpt-4o"
    mock_cfg.app_name = "test_helpdesk"
    mock_cfg.log_level = "DEBUG"
    mock_cfg.upstash_redis_rest_url = "https://mock.upstash.io"
    mock_cfg.upstash_redis_rest_token = "test_token"
    mock_cfg.allowed_origins = ["*"]
    with patch(
        "src.customer_helpdesk.config.get_settings",
        return_value=mock_cfg,
    ):
        yield mock_cfg


class TestSessionPersistence:
    """Tests for session persistence operations."""

    @pytest.mark.asyncio
    async def test_create_session_persists_state(self, mock_settings):
        """Test that session state is persisted correctly on create."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
            Session,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_session.execute.return_value = None
            mock_session.commit.return_value = None
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True
            state = {"user_id": "user123", "classifier_intent": "billing"}

            result = await service.create_session(
                app_name="test_app",
                user_id="user123",
                session_id="session456",
                state=state,
            )

            assert isinstance(result, Session)
            assert result.app_name == "test_app"
            assert result.user_id == "user123"
            assert result.session_id == "session456"
            assert result.state == state

    @pytest.mark.asyncio
    async def test_get_session_returns_persisted_state(self, mock_settings):
        """Test that get_session returns the persisted session state."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
            Session,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            now = MagicMock()
            mock_result.fetchone.return_value = (
                "test_app",
                "user123",
                "session456",
                '{"user_id": "user123", "preference": "dark_mode"}',
                now,
                now,
            )
            mock_session.execute.return_value = mock_result
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True

            result = await service.get_session(
                app_name="test_app",
                user_id="user123",
                session_id="session456",
            )

            assert result is not None
            assert isinstance(result, Session)
            assert result.app_name == "test_app"
            assert result.user_id == "user123"
            assert result.session_id == "session456"
            assert result.state == {"user_id": "user123", "preference": "dark_mode"}

    @pytest.mark.asyncio
    async def test_get_session_returns_none_for_nonexistent(self, mock_settings):
        """Test that get_session returns None for non-existent session."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = None
            mock_session.execute.return_value = mock_result
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True

            result = await service.get_session(
                app_name="test_app",
                user_id="nonexistent",
                session_id="session999",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_user_id_stored_in_session_state(self, mock_settings):
        """Test that user_id is stored in session state correctly."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_session.execute.return_value = None
            mock_session.commit.return_value = None
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True
            user_id = "user_abc_123"
            state = {"user_id": user_id, "action": "browse"}

            result = await service.create_session(
                app_name="helpdesk",
                user_id=user_id,
                session_id="sess_001",
                state=state,
            )

            assert result.state["user_id"] == user_id
            assert result.user_id == user_id


class TestSafeJsonParseIntegration:
    """Tests for safe_json_parse integration with session service."""

    @pytest.mark.asyncio
    async def test_get_session_handles_corrupted_json(self, mock_settings):
        """Test that safe_json_parse handles corrupted JSONB gracefully."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            now = MagicMock()
            corrupted_json = '{"user_id": "user123", "broken": '
            mock_result.fetchone.return_value = (
                "test_app",
                "user123",
                "session456",
                corrupted_json,
                now,
                now,
            )
            mock_session.execute.return_value = mock_result
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True

            result = await service.get_session(
                app_name="test_app",
                user_id="user123",
                session_id="session456",
            )

            assert result is not None
            assert result.state == {}

    @pytest.mark.asyncio
    async def test_list_sessions_handles_corrupted_json(self, mock_settings):
        """Test that list_sessions handles corrupted JSONB in any row gracefully."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            now = MagicMock()
            mock_result.fetchall.return_value = [
                (
                    "test_app",
                    "user123",
                    "session1",
                    '{"intent": "billing"}',
                    now,
                    now,
                ),
                (
                    "test_app",
                    "user123",
                    "session2",
                    '{"intent": "technical',
                    now,
                    now,
                ),
            ]
            mock_session.execute.return_value = mock_result
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True

            results = await service.list_sessions(
                app_name="test_app",
                user_id="user123",
            )

            assert len(results) == 2
            assert results[0].state == {"intent": "billing"}
            assert results[1].state == {}


class TestSessionStateValidation:
    """Tests for session state validation integration."""

    @pytest.mark.asyncio
    async def test_session_state_with_various_data_types(self, mock_settings):
        """Test session state persistence with various data types."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_session.execute.return_value = None
            mock_session.commit.return_value = None
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True
            state = {
                "user_id": "user123",
                "count": 42,
                "active": True,
                "tags": ["billing", "urgent"],
                "metadata": {"priority": "high", "score": 9.5},
                "nil": None,
            }

            result = await service.create_session(
                app_name="test_app",
                user_id="user123",
                session_id="session_diverse",
                state=state,
            )

            assert result.state == state
            assert result.state["count"] == 42
            assert result.state["active"] is True
            assert result.state["tags"] == ["billing", "urgent"]
            assert result.state["metadata"] == {"priority": "high", "score": 9.5}
            assert result.state["nil"] is None

    @pytest.mark.asyncio
    async def test_update_session_state_validation(self, mock_settings):
        """Test that update_session_state properly validates and persists state."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.rowcount = 1
            mock_session.execute.return_value = mock_result
            mock_session.commit.return_value = None
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True
            new_state = {"validation": "passed", "version": 2}

            result = await service.update_session_state(
                app_name="test_app",
                user_id="user123",
                session_id="session456",
                new_state=new_state,
            )

            assert result is not None
            assert result.state == new_state
            assert result.app_name == "test_app"
            assert result.user_id == "user123"
            assert result.session_id == "session456"

    @pytest.mark.asyncio
    async def test_update_session_state_returns_none_for_missing(self, mock_settings):
        """Test that update_session_state returns None for non-existent session."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.rowcount = 0
            mock_session.execute.return_value = mock_result
            mock_session.commit.return_value = None
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True

            result = await service.update_session_state(
                app_name="nonexistent",
                user_id="user999",
                session_id="session_missing",
                new_state={"test": "data"},
            )

            assert result is None


class TestSessionOperationsEndToEnd:
    """End-to-end style tests for session operations without actual DB."""

    @pytest.mark.asyncio
    async def test_create_and_get_cycle(self, mock_settings):
        """Test complete create-then-get cycle for session persistence."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_session.execute.return_value = MagicMock()
            mock_session.commit.return_value = None
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True
            original_state = {"user_id": "user_xyz", "flow": "create_get_test"}

            created = await service.create_session(
                app_name="app_test",
                user_id="user_xyz",
                session_id="sess_test",
                state=original_state,
            )

            assert created is not None
            assert created.state == original_state

    @pytest.mark.asyncio
    async def test_delete_session_returns_true_on_success(self, mock_settings):
        """Test that delete_session returns True when session is deleted."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.rowcount = 1
            mock_session.execute.return_value = mock_result
            mock_session.commit.return_value = None
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True

            result = await service.delete_session(
                app_name="test_app",
                user_id="user123",
                session_id="session_delete",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_delete_session_returns_false_when_not_found(self, mock_settings):
        """Test that delete_session returns False when session doesn't exist."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.rowcount = 0
            mock_session.execute.return_value = mock_result
            mock_session.commit.return_value = None
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True

            result = await service.delete_session(
                app_name="nonexistent",
                user_id="user999",
                session_id="session_missing",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_list_sessions_returns_all_user_sessions(self, mock_settings):
        """Test that list_sessions returns all sessions for a user."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            now = MagicMock()
            mock_result.fetchall.return_value = [
                ("app1", "user1", "sess1", '{"topic": "billing"}', now, now),
                ("app1", "user1", "sess2", '{"topic": "technical"}', now, now),
                ("app1", "user1", "sess3", '{"topic": "general"}', now, now),
            ]
            mock_session.execute.return_value = mock_result
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True

            results = await service.list_sessions(
                app_name="app1",
                user_id="user1",
            )

            assert len(results) == 3
            assert results[0].session_id == "sess1"
            assert results[1].session_id == "sess2"
            assert results[2].session_id == "sess3"
            assert results[0].state == {"topic": "billing"}
            assert results[1].state == {"topic": "technical"}
            assert results[2].state == {"topic": "general"}

    @pytest.mark.asyncio
    async def test_list_sessions_returns_empty_list_when_none(self, mock_settings):
        """Test that list_sessions returns empty list when no sessions exist."""
        from src.customer_helpdesk.services.session_service import (
            DatabaseSessionService,
        )

        with patch(
            "src.customer_helpdesk.services.session_service.async_session_factory"
        ) as mock_factory:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchall.return_value = []
            mock_session.execute.return_value = mock_result
            mock_factory.return_value.__aenter__.return_value = mock_session

            service = DatabaseSessionService()
            service._tables_created = True

            results = await service.list_sessions(
                app_name="test_app",
                user_id="new_user",
            )

            assert isinstance(results, list)
            assert len(results) == 0
