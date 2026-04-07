import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from httpx import ASGITransport, AsyncClient
import sys

from customer_helpdesk.models.errors import ErrorCode, ErrorResponse


@pytest.fixture(autouse=True)
def clear_customer_helpdesk_modules():
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("customer_helpdesk"):
            del sys.modules[mod_name]
    yield


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.app_name = "test_helpdesk"
    settings.allowed_origins = ["*"]
    settings.database_url = "postgresql+asyncpg://test:test@localhost/test"
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_collection = "test_collection"
    settings.openai_api_key = "sk-test"
    settings.basic_model = "openai/gpt-4o-mini"
    settings.complex_model = "openai/gpt-4o"
    settings.log_level = "DEBUG"
    settings.upstash_redis_rest_url = "https://mock.upstash.io"
    settings.upstash_redis_rest_token = "test_token"
    return settings


@pytest.fixture
def mock_session_service():
    mock = MagicMock()
    mock.get_session = AsyncMock(return_value=None)
    mock.create_session = AsyncMock(return_value=MagicMock())
    return mock


@pytest.mark.asyncio
async def test_error_response_structure():
    error = ErrorResponse(
        error_code=ErrorCode.AGENT_ERROR,
        message="Test error message",
        correlation_id="abc12345",
    )

    assert error.error_code == "AGENT_ERROR"
    assert error.message == "Test error message"
    assert error.correlation_id == "abc12345"


@pytest.mark.asyncio
async def test_error_response_serialization():
    error = ErrorResponse(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Validation failed",
        correlation_id="xyz789",
    )

    dumped = error.model_dump()

    assert dumped["error_code"] == "VALIDATION_ERROR"
    assert dumped["message"] == "Validation failed"
    assert dumped["correlation_id"] == "xyz789"
    assert "extra" not in dumped


@pytest.mark.asyncio
async def test_error_codes_enum():
    assert ErrorCode.AGENT_ERROR.value == "AGENT_ERROR"
    assert ErrorCode.VALIDATION_ERROR.value == "VALIDATION_ERROR"
    assert ErrorCode.SESSION_ERROR.value == "SESSION_ERROR"
    assert ErrorCode.INTERNAL_ERROR.value == "INTERNAL_ERROR"
    assert len(ErrorCode) == 4


@pytest.mark.asyncio
async def test_error_response_without_correlation_id():
    error = ErrorResponse(
        error_code=ErrorCode.INTERNAL_ERROR, message="Internal server error"
    )

    assert error.correlation_id is None
    dumped = error.model_dump()
    assert dumped["correlation_id"] is None


@pytest.mark.asyncio
async def test_error_response_extra_forbidden():
    with pytest.raises(ValueError):
        ErrorResponse(
            error_code=ErrorCode.AGENT_ERROR,
            message="Test",
            correlation_id="123",
            extra_field="should fail",
        )


@pytest.mark.asyncio
async def test_chat_endpoint_error_returns_structured_response(
    mock_settings, mock_session_service
):
    with patch("customer_helpdesk.config.get_settings", return_value=mock_settings):
        with patch("customer_helpdesk.main.get_settings", return_value=mock_settings):
            with patch(
                "customer_helpdesk.main.DatabaseSessionService",
                return_value=mock_session_service,
            ):
                with patch("customer_helpdesk.main.Runner") as mock_runner_class:
                    mock_runner = MagicMock()
                    mock_runner_class.return_value = mock_runner

                    async def raise_error(*args, **kwargs):
                        raise RuntimeError("Database connection failed")

                    mock_runner.run_async = raise_error

                    from customer_helpdesk.main import app

                    async with AsyncClient(
                        transport=ASGITransport(app=app), base_url="http://test"
                    ) as client:
                        response = await client.post(
                            "/chat",
                            json={
                                "messages": [{"role": "user", "content": "Hello"}],
                                "user_id": "test_user",
                            },
                        )

                    assert response.status_code == 500
                    data = response.json()

                    assert "error_code" in data
                    assert "message" in data
                    assert "correlation_id" in data

                    assert data["error_code"] == "AGENT_ERROR"
                    assert "Database connection failed" not in data["message"]
                    assert len(data["correlation_id"]) == 8


@pytest.mark.asyncio
async def test_error_response_internal_details_not_exposed(
    mock_settings, mock_session_service
):
    error_message = "Sensitive stack trace: /path/to/file.py line 42"

    with patch("customer_helpdesk.config.get_settings", return_value=mock_settings):
        with patch("customer_helpdesk.main.get_settings", return_value=mock_settings):
            with patch(
                "customer_helpdesk.main.DatabaseSessionService",
                return_value=mock_session_service,
            ):
                with patch("customer_helpdesk.main.Runner") as mock_runner_class:
                    mock_runner = MagicMock()
                    mock_runner_class.return_value = mock_runner

                    async def raise_error(*args, **kwargs):
                        raise ValueError(error_message)

                    mock_runner.run_async = raise_error

                    from customer_helpdesk.main import app

                    async with AsyncClient(
                        transport=ASGITransport(app=app), base_url="http://test"
                    ) as client:
                        response = await client.post(
                            "/chat",
                            json={
                                "messages": [{"role": "user", "content": "Hello"}],
                                "user_id": "test_user",
                            },
                        )

                    assert response.status_code == 500
                    data = response.json()

                    assert error_message not in data["message"]
                    assert "Sensitive stack trace" not in data["message"]
                    assert "line 42" not in data["message"]
