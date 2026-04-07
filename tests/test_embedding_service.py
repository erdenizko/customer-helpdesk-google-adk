"""Unit tests for embedding service."""

from unittest.mock import AsyncMock, patch

import pytest

from src.customer_helpdesk.services.embedding_service import generate_embedding


class TestGenerateEmbedding:
    """Tests for generate_embedding function."""

    @pytest.mark.asyncio
    async def test_returns_correct_dimension(self):
        """Test that embedding vector has 1536 dimensions."""
        expected_dimension = 1536
        mock_embedding = [0.1] * expected_dimension

        with patch(
            "src.customer_helpdesk.services.embedding_service.litellm"
        ) as mock_litellm:
            mock_litellm.aembedding = AsyncMock(
                return_value={"data": [{"embedding": mock_embedding}]}
            )
            result = await generate_embedding("test text")

            assert len(result) == expected_dimension
            assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_determinism_same_text(self):
        """Test that same input text produces identical output vector."""
        mock_embedding = [0.5] * 1536

        with patch(
            "src.customer_helpdesk.services.embedding_service.litellm"
        ) as mock_litellm:
            mock_litellm.aembedding = AsyncMock(
                return_value={"data": [{"embedding": mock_embedding}]}
            )

            result1 = await generate_embedding("hello world")
            result2 = await generate_embedding("hello world")

            assert result1 == result2

    @pytest.mark.asyncio
    async def test_normalization_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        mock_embedding = [0.3] * 1536

        with patch(
            "src.customer_helpdesk.services.embedding_service.litellm"
        ) as mock_litellm:
            mock_litellm.aembedding = AsyncMock(
                return_value={"data": [{"embedding": mock_embedding}]}
            )

            await generate_embedding("  hello world  ")
            await generate_embedding("hello world")

            calls = mock_litellm.aembedding.call_args_list
            for call in calls:
                normalized_input = call.kwargs["input"][0]
                assert normalized_input == "hello world"

    @pytest.mark.asyncio
    async def test_normalization_lowercase(self):
        """Test that text is converted to lowercase before embedding."""
        mock_embedding = [0.2] * 1536

        with patch(
            "src.customer_helpdesk.services.embedding_service.litellm"
        ) as mock_litellm:
            mock_litellm.aembedding = AsyncMock(
                return_value={"data": [{"embedding": mock_embedding}]}
            )

            await generate_embedding("HELLO WORLD")
            await generate_embedding("Hello World")

            calls = mock_litellm.aembedding.call_args_list
            for call in calls:
                normalized_input = call.kwargs["input"][0]
                assert normalized_input == "hello world"

    @pytest.mark.asyncio
    async def test_error_handling_litellm_failure(self):
        """Test that LiteLLM failures are propagated."""
        with patch(
            "src.customer_helpdesk.services.embedding_service.litellm"
        ) as mock_litellm:
            mock_litellm.aembedding = AsyncMock(
                side_effect=Exception("LiteLLM API error")
            )

            with pytest.raises(Exception, match="LiteLLM API error"):
                await generate_embedding("test text")

    @pytest.mark.asyncio
    async def test_custom_model(self):
        """Test that custom model is passed to LiteLLM."""
        custom_model = "text-embedding-3-small"
        mock_embedding = [0.1] * 1536

        with patch(
            "src.customer_helpdesk.services.embedding_service.litellm"
        ) as mock_litellm:
            mock_litellm.aembedding = AsyncMock(
                return_value={"data": [{"embedding": mock_embedding}]}
            )

            await generate_embedding("test", model=custom_model)

            mock_litellm.aembedding.assert_called_once()
            call_kwargs = mock_litellm.aembedding.call_args.kwargs
            assert call_kwargs["model"] == custom_model
