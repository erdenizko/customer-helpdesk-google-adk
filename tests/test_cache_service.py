"""Unit tests for CacheService."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from customer_helpdesk.services.cache_service import CacheService


@pytest.fixture
def mock_redis():
    """Create a mock Upstash Redis client."""
    mock = MagicMock()
    mock.get = AsyncMock()
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    return mock


@pytest.fixture
def cache_service(mock_redis):
    """Create CacheService with mocked Redis."""
    with patch(
        "customer_helpdesk.services.cache_service.Redis", return_value=mock_redis
    ):
        service = CacheService(
            redis_url="https://mock.upstash.io",
            redis_token="test_token",
            key_prefix="test",
        )
        service._redis = mock_redis
    return service


class TestCacheServiceGet:
    """Tests for CacheService.get method."""

    async def test_get_returns_cached_value(self, cache_service, mock_redis):
        """Should return parsed JSON dict when key exists."""
        mock_redis.get.return_value = '{"user_id": 123, "name": "Alice"}'

        result = await cache_service.get("user", "alice")

        assert result == {"user_id": 123, "name": "Alice"}
        mock_redis.get.assert_called_once_with("v1:test:user:alice")

    async def test_get_returns_none_when_not_found(self, cache_service, mock_redis):
        """Should return None when key does not exist."""
        mock_redis.get.return_value = None

        result = await cache_service.get("user", "nonexistent")

        assert result is None

    async def test_get_returns_none_on_redis_error(self, cache_service, mock_redis):
        """Should return None (fail-open) on Redis errors."""
        mock_redis.get.side_effect = Exception("Connection refused")

        result = await cache_service.get("user", "alice")

        assert result is None

    async def test_get_handles_non_string_result(self, cache_service, mock_redis):
        """Should handle non-string results (e.g., from other Redis operations)."""
        mock_redis.get.return_value = {
            "user_id": 123
        }  # Already a dict, not JSON string

        result = await cache_service.get("user", "alice")

        assert result == {"user_id": 123}


class TestCacheServiceSet:
    """Tests for CacheService.set method."""

    async def test_set_returns_true_on_success(self, cache_service, mock_redis):
        """Should return True when set succeeds."""
        mock_redis.set.return_value = True

        result = await cache_service.set("user", "alice", {"name": "Alice"}, ttl=3600)

        assert result is True
        mock_redis.set.assert_called_once()
        call_kwargs = mock_redis.set.call_args.kwargs
        assert call_kwargs["ex"] == 3600

    async def test_set_returns_true_when_result_is_ok(self, cache_service, mock_redis):
        """Should return True when Redis returns 'OK'."""
        mock_redis.set.return_value = "OK"

        result = await cache_service.set("user", "alice", {"name": "Alice"}, ttl=3600)

        assert result is True

    async def test_set_returns_false_on_error(self, cache_service, mock_redis):
        """Should return False (fail-open) on Redis errors."""
        mock_redis.set.side_effect = Exception("Connection refused")

        result = await cache_service.set("user", "alice", {"name": "Alice"}, ttl=3600)

        assert result is False

    async def test_set_serializes_value_to_json(self, cache_service, mock_redis):
        """Should serialize dict value to JSON string."""
        mock_redis.set.return_value = True

        await cache_service.set("user", "alice", {"name": "Alice", "age": 30}, ttl=3600)

        call_args = mock_redis.set.call_args
        import json

        stored_value = json.loads(call_args[0][1])  # Second positional arg is value
        assert stored_value == {"name": "Alice", "age": 30}


class TestCacheServiceDelete:
    """Tests for CacheService.delete method."""

    async def test_delete_returns_true_when_deleted(self, cache_service, mock_redis):
        """Should return True when key was deleted."""
        mock_redis.delete.return_value = 1

        result = await cache_service.delete("user", "alice")

        assert result is True
        mock_redis.delete.assert_called_once_with("v1:test:user:alice")

    async def test_delete_returns_false_when_not_found(self, cache_service, mock_redis):
        """Should return False when key does not exist."""
        mock_redis.delete.return_value = 0

        result = await cache_service.delete("user", "nonexistent")

        assert result is False

    async def test_delete_returns_false_on_error(self, cache_service, mock_redis):
        """Should return False (fail-open) on Redis errors."""
        mock_redis.delete.side_effect = Exception("Connection refused")

        result = await cache_service.delete("user", "alice")

        assert result is False


class TestCacheServiceGetOrSet:
    """Tests for CacheService.get_or_set method."""

    async def test_get_or_set_returns_cached_value(self, cache_service, mock_redis):
        """Should return cached value without calling factory."""
        mock_redis.get.return_value = '{"user_id": 123}'

        factory = AsyncMock(return_value={"user_id": 999})
        result = await cache_service.get_or_set("user", "alice", factory, ttl=3600)

        assert result == {"user_id": 123}
        factory.assert_not_called()

    async def test_get_or_set_computes_and_caches_when_not_found(
        self, cache_service, mock_redis
    ):
        """Should compute value via factory and cache it when key missing."""
        mock_redis.get.side_effect = [
            None,
            None,
            '{"user_id": 123}',
        ]  # Initial + after lock + after set
        mock_redis.set.side_effect = [True, True]  # Lock + cache set

        factory = AsyncMock(return_value={"user_id": 123})
        result = await cache_service.get_or_set("user", "alice", factory, ttl=3600)

        assert result == {"user_id": 123}
        factory.assert_called_once()

    async def test_get_or_set_returns_none_when_factory_returns_none(
        self, cache_service, mock_redis
    ):
        """Should return None when factory returns None."""
        mock_redis.get.return_value = None

        factory = AsyncMock(return_value=None)
        result = await cache_service.get_or_set("user", "alice", factory, ttl=3600)

        assert result is None
        mock_redis.set.assert_not_called()

    async def test_get_or_set_prevents_stampede_with_lock(
        self, cache_service, mock_redis
    ):
        """Should acquire lock to prevent stampede when computing value."""
        # First get returns None (cache miss)
        # Lock acquisition succeeds
        # Second get (after lock) returns None
        # Set succeeds
        mock_redis.get.side_effect = [None, None, '{"user_id": 123}']
        mock_redis.set.side_effect = [True, True]  # Lock acquired, then cache set

        factory = AsyncMock(return_value={"user_id": 123})
        result = await cache_service.get_or_set("user", "alice", factory, ttl=3600)

        assert result == {"user_id": 123}
        # Verify lock was acquired with NX and EX
        lock_call = mock_redis.set.call_args_list[0]
        lock_key = lock_call[0][0]
        assert "lock" in lock_key
        assert lock_call[1]["nx"] is True
        assert lock_call[1]["ex"] <= 30  # Lock TTL capped at 30

    async def test_get_or_set_retries_cache_after_lock_failure(
        self, cache_service, mock_redis
    ):
        """Should retry cache after failing to acquire lock (another request computing)."""
        # First get returns None
        # Factory computes value
        # Lock acquisition fails (another request has lock)
        # Retry get returns the computed value (populated by other request)
        mock_redis.get.side_effect = [None, '{"user_id": 123}', '{"user_id": 123}']
        mock_redis.set.side_effect = [False, True]  # Lock fails, cache set

        factory = AsyncMock(return_value={"user_id": 999})
        result = await cache_service.get_or_set("user", "alice", factory, ttl=3600)

        # Factory WAS called (it's called before lock acquisition per code flow)
        factory.assert_called_once()
        # But we got the cached value from the retry, not the computed one
        assert result == {"user_id": 123}


class TestCacheServiceKeyVersioning:
    """Tests for key versioning."""

    def test_make_key_includes_version(self, cache_service):
        """Should include version prefix in keys."""
        key = cache_service._make_key("user", "alice")
        assert key.startswith("v1:test:user:alice")

    def test_make_key_with_different_prefixes(self):
        """Should include key_prefix in keys."""
        with patch("customer_helpdesk.services.cache_service.Redis"):
            service1 = CacheService(
                redis_url="https://mock.upstash.io",
                redis_token="test_token",
                key_prefix="cache1",
            )
            service2 = CacheService(
                redis_url="https://mock.upstash.io",
                redis_token="test_token",
                key_prefix="cache2",
            )

        assert "cache1" in service1._make_key("user", "alice")
        assert "cache2" in service2._make_key("user", "alice")


class TestCacheServiceNegativeCache:
    """Tests for negative caching."""

    async def test_set_negative_caches_error(self, cache_service, mock_redis):
        """Should store error in negative cache with TTL."""
        mock_redis.set.return_value = True

        result = await cache_service.set_negative("user", "alice", "User not found")

        assert result is True
        call_args = mock_redis.set.call_args
        import json

        stored_value = json.loads(call_args[0][1])
        assert stored_value["error"] == "User not found"
        assert "timestamp" in stored_value
        # Check negative cache TTL
        assert call_args[1]["ex"] == 30

    async def test_get_negative_returns_error_dict(self, cache_service, mock_redis):
        """Should return error dict when negative cache hit."""
        import time

        timestamp = time.time()
        mock_redis.get.return_value = (
            f'{{"error": "User not found", "timestamp": {timestamp}}}'
        )

        result = await cache_service.get_negative("user", "alice")

        assert result["error"] == "User not found"
        assert result["timestamp"] == timestamp

    async def test_get_negative_returns_none_on_miss(self, cache_service, mock_redis):
        """Should return None when negative cache miss."""
        mock_redis.get.return_value = None

        result = await cache_service.get_negative("user", "alice")

        assert result is None

    async def test_negative_cache_uses_hashed_key(self, cache_service):
        """Should use SHA256 hashed key for negative cache."""
        neg_key = cache_service._make_negative_key("user", "alice")
        original_key = cache_service._make_key("user", "alice")

        import hashlib

        expected_hash = hashlib.sha256(original_key.encode()).hexdigest()
        assert neg_key == f"v1:neg:{expected_hash}"
        assert "alice" not in neg_key  # Original key should not be in negative key


class TestCacheServiceFailOpen:
    """Tests for fail-open behavior on Redis errors."""

    async def test_get_fail_open(self, cache_service, mock_redis):
        """get() should return None on Redis error."""
        mock_redis.get.side_effect = Exception("Redis unavailable")
        assert await cache_service.get("user", "alice") is None

    async def test_set_fail_open(self, cache_service, mock_redis):
        """set() should return False on Redis error."""
        mock_redis.set.side_effect = Exception("Redis unavailable")
        assert await cache_service.set("user", "alice", {}, 3600) is False

    async def test_delete_fail_open(self, cache_service, mock_redis):
        """delete() should return False on Redis error."""
        mock_redis.delete.side_effect = Exception("Redis unavailable")
        assert await cache_service.delete("user", "alice") is False

    async def test_set_negative_fail_open(self, cache_service, mock_redis):
        """set_negative() should return False on Redis error."""
        mock_redis.set.side_effect = Exception("Redis unavailable")
        assert await cache_service.set_negative("user", "alice", "error") is False

    async def test_get_negative_fail_open(self, cache_service, mock_redis):
        """get_negative() should return None on Redis error."""
        mock_redis.get.side_effect = Exception("Redis unavailable")
        assert await cache_service.get_negative("user", "alice") is None


class TestCacheServiceEdgeCases:
    """Tests for edge cases."""

    async def test_get_with_empty_key(self, cache_service, mock_redis):
        """Should handle empty key strings."""
        mock_redis.get.return_value = '{"data": "test"}'

        result = await cache_service.get("user", "")

        assert result == {"data": "test"}
        mock_redis.get.assert_called_once()

    async def test_set_with_none_value(self, cache_service, mock_redis):
        """Should handle None values in dict."""
        mock_redis.set.return_value = True

        result = await cache_service.set("user", "alice", {"name": None}, ttl=3600)

        assert result is True
        import json

        call_args = mock_redis.set.call_args
        stored = json.loads(call_args[0][1])
        assert stored == {"name": None}

    async def test_set_with_expiry_boundary(self, cache_service, mock_redis):
        """Should handle TTL boundary values."""
        mock_redis.set.return_value = True

        # Very short TTL
        await cache_service.set("user", "alice", {"data": "test"}, ttl=1)
        assert mock_redis.set.call_args.kwargs["ex"] == 1

        # Very long TTL
        await cache_service.set("user", "bob", {"data": "test"}, ttl=86400)
        assert mock_redis.set.call_args.kwargs["ex"] == 86400

    async def test_get_or_set_lock_ttl_capped_at_30(self, cache_service, mock_redis):
        """Lock TTL should be capped at 30 seconds even if cache TTL is longer."""
        mock_redis.get.side_effect = [None, None, '{"result": "data"}']
        mock_redis.set.side_effect = [True, True]

        factory = AsyncMock(return_value={"result": "data"})
        await cache_service.get_or_set("user", "alice", factory, ttl=3600)

        # First set call is for the lock
        lock_call = mock_redis.set.call_args_list[0]
        assert lock_call[1]["ex"] == 30  # Capped at 30

    async def test_get_or_set_lock_ttl_uses_actual_ttl_if_smaller(
        self, cache_service, mock_redis
    ):
        """Lock TTL should use cache TTL if it's smaller than 30."""
        mock_redis.get.side_effect = [None, None, '{"result": "data"}']
        mock_redis.set.side_effect = [True, True]

        factory = AsyncMock(return_value={"result": "data"})
        await cache_service.get_or_set("user", "alice", factory, ttl=10)

        lock_call = mock_redis.set.call_args_list[0]
        assert lock_call[1]["ex"] == 10  # Uses actual TTL since < 30
