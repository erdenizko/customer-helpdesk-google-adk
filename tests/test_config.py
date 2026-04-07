import pytest
from pydantic import ValidationError

from src.customer_helpdesk.config import Settings


class TestAllowedOrigins:
    """Tests for CORS allowed_origins configuration."""

    def test_default_allowed_origins_is_wildcard(self):
        """Default CORS should allow all origins when not configured."""
        settings = Settings(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            qdrant_url="http://localhost:6333",
            qdrant_collection="test_collection",
            openai_api_key="sk-test",
            basic_model="openai/gpt-4o-mini",
            complex_model="openai/gpt-4o",
            app_name="test_helpdesk",
            log_level="DEBUG",
            upstash_redis_rest_url="https://mock.upstash.io",
            upstash_redis_rest_token="test_token",
        )
        assert settings.allowed_origins == ["*"]

    def test_allowed_origins_from_comma_separated_string(self):
        """Test parsing comma-separated string into list of origins."""
        settings = Settings(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            qdrant_url="http://localhost:6333",
            qdrant_collection="test_collection",
            openai_api_key="sk-test",
            basic_model="openai/gpt-4o-mini",
            complex_model="openai/gpt-4o",
            app_name="test_helpdesk",
            log_level="DEBUG",
            upstash_redis_rest_url="https://mock.upstash.io",
            upstash_redis_rest_token="test_token",
            allowed_origins="https://example.com,https://app.example.com",
        )
        assert settings.allowed_origins == [
            "https://example.com",
            "https://app.example.com",
        ]

    def test_allowed_origins_with_spaces_trimmed(self):
        """Test that spaces around origins are trimmed."""
        settings = Settings(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            qdrant_url="http://localhost:6333",
            qdrant_collection="test_collection",
            openai_api_key="sk-test",
            basic_model="openai/gpt-4o-mini",
            complex_model="openai/gpt-4o",
            app_name="test_helpdesk",
            log_level="DEBUG",
            upstash_redis_rest_url="https://mock.upstash.io",
            upstash_redis_rest_token="test_token",
            allowed_origins=" https://example.com , https://app.example.com ",
        )
        assert settings.allowed_origins == [
            "https://example.com",
            "https://app.example.com",
        ]

    def test_allowed_origins_explicit_list(self):
        """Test that explicit list is preserved."""
        settings = Settings(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            qdrant_url="http://localhost:6333",
            qdrant_collection="test_collection",
            openai_api_key="sk-test",
            basic_model="openai/gpt-4o-mini",
            complex_model="openai/gpt-4o",
            app_name="test_helpdesk",
            log_level="DEBUG",
            upstash_redis_rest_url="https://mock.upstash.io",
            upstash_redis_rest_token="test_token",
            allowed_origins=["https://only-this.com"],
        )
        assert settings.allowed_origins == ["https://only-this.com"]

    def test_allowed_origins_empty_string_becomes_empty_list(self):
        """Test that empty string results in empty list."""
        settings = Settings(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            qdrant_url="http://localhost:6333",
            qdrant_collection="test_collection",
            openai_api_key="sk-test",
            basic_model="openai/gpt-4o-mini",
            complex_model="openai/gpt-4o",
            app_name="test_helpdesk",
            log_level="DEBUG",
            upstash_redis_rest_url="https://mock.upstash.io",
            upstash_redis_rest_token="test_token",
            allowed_origins="",
        )
        assert settings.allowed_origins == []
