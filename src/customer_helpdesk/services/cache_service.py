"""Upstash Redis HTTP-based cache service with distributed lock support."""

import hashlib
import json
import time
import uuid
from typing import Any, Callable, Optional

from upstash_redis.asyncio import Redis


class CacheService:
    """
    Async cache service using Upstash Redis HTTP REST API.

    Provides get/set/delete operations with JSON serialization,
    distributed locking for stampede prevention, and fail-open error handling.

    Args:
        redis_url: Upstash Redis REST API URL
        redis_token: Upstash Redis REST API token
        key_prefix: Optional prefix for all keys (default: "v1")

    Note:
        Uses HTTP REST API (not TCP) for serverless compatibility.
        Fails open: returns None on Redis errors instead of raising exceptions.
    """

    VERSION = "v1"
    NEG_PREFIX = "neg"
    NEGATIVE_CACHE_TTL = 30

    def __init__(
        self,
        redis_url: str,
        redis_token: str,
        key_prefix: str = "cache",
    ):
        self._redis = Redis(url=redis_url, token=redis_token)
        self._key_prefix = key_prefix

    def _make_key(self, key_type: str, key: str) -> str:
        """Build a versioned key: v1:{key_prefix}:{key_type}:{key}."""
        return f"{self.VERSION}:{self._key_prefix}:{key_type}:{key}"

    async def get(self, key_type: str, key: str) -> Optional[dict]:
        """
        Async get with JSON parse.

        Args:
            key_type: Category of the key (e.g., "session", "user")
            key: The cache key

        Returns:
            Parsed JSON dict if found, None if not found or on error (fail-open)
        """
        full_key = self._make_key(key_type, key)
        try:
            result = await self._redis.get(full_key)
            if result is None:
                return None
            if isinstance(result, str):
                return json.loads(result)
            return result
        except Exception:
            # Fail-open: return None on any Redis error
            return None

    async def set(
        self,
        key_type: str,
        key: str,
        value: dict,
        ttl: int,
    ) -> bool:
        """
        Async set with TTL.

        Args:
            key_type: Category of the key
            key: The cache key
            value: Dict value to serialize as JSON
            ttl: Time-to-live in seconds

        Returns:
            True if set succeeded, False on error (fail-open)
        """
        full_key = self._make_key(key_type, key)
        try:
            serialized = json.dumps(value)
            result = await self._redis.set(full_key, serialized, ex=ttl)
            return result == "OK" or result is True
        except Exception:
            # Fail-open: return False on any Redis error
            return False

    async def delete(self, key_type: str, key: str) -> bool:
        """
        Async delete.

        Args:
            key_type: Category of the key
            key: The cache key

        Returns:
            True if deleted, False if not found or on error (fail-open)
        """
        full_key = self._make_key(key_type, key)
        try:
            result = await self._redis.delete(full_key)
            return result > 0
        except Exception:
            # Fail-open: return False on any Redis error
            return False

    async def get_or_set(
        self,
        key_type: str,
        key: str,
        factory: Callable[[], Any],
        ttl: int,
    ) -> Optional[dict]:
        """
        Get from cache or compute and cache.

        Uses distributed lock (setnx pattern) to prevent stampede
        when multiple requests race to compute the same key.

        Args:
            key_type: Category of the key
            key: The cache key
            factory: Async callable that computes the value if not cached
            ttl: Time-to-live in seconds for the cached value

        Returns:
            Cached or computed value as dict, or None on error
        """
        # Try to get from cache first
        cached = await self.get(key_type, key)
        if cached is not None:
            return cached

        # Compute value (may be slow)
        value = await factory()
        if value is None:
            return None

        # Try to acquire distributed lock to prevent stampede
        lock_key = self._make_key("lock", f"{key_type}:{key}")
        lock_value = str(uuid.uuid4())
        lock_ttl = min(ttl, 30)  # Lock TTL should not exceed cache TTL

        try:
            # SET NX EX pattern for distributed lock
            lock_acquired = await self._redis.set(
                lock_key, lock_value, nx=True, ex=lock_ttl
            )
        except Exception:
            # Fail-open: if lock acquisition fails, still try to set cache
            lock_acquired = False

        if not lock_acquired:
            # Another request is computing this key, wait and retry cache
            import asyncio

            await asyncio.sleep(0.5)
            cached = await self.get(key_type, key)
            if cached is not None:
                return cached
            # Fall through to compute and set anyway if still not cached

        # Double-check cache after lock acquisition
        cached = await self.get(key_type, key)
        if cached is not None:
            return cached

        # Set computed value in cache
        await self.set(key_type, key, value, ttl)

        return value if isinstance(value, dict) else None

    def _make_negative_key(self, key_type: str, key: str) -> str:
        original_key = self._make_key(key_type, key)
        hashed = hashlib.sha256(original_key.encode()).hexdigest()
        return f"{self.VERSION}:{self.NEG_PREFIX}:{hashed}"

    async def set_negative(
        self,
        key_type: str,
        key: str,
        error_message: str,
    ) -> bool:
        """
        Store an error in negative cache with 30s TTL.

        Args:
            key_type: Category of the key
            key: The cache key
            error_message: Error message to cache

        Returns:
            True if set succeeded, False on error (fail-open)
        """
        neg_key = self._make_negative_key(key_type, key)
        value = {
            "error": error_message,
            "timestamp": time.time(),
        }
        try:
            serialized = json.dumps(value)
            result = await self._redis.set(
                neg_key, serialized, ex=self.NEGATIVE_CACHE_TTL
            )
            return result == "OK" or result is True
        except Exception:
            return False

    async def get_negative(
        self,
        key_type: str,
        key: str,
    ) -> Optional[dict]:
        """
        Check if a key has a cached error.

        Args:
            key_type: Category of the key
            key: The cache key

        Returns:
            Dict with 'error' and 'timestamp' if negative cache hit, None otherwise
        """
        neg_key = self._make_negative_key(key_type, key)
        try:
            result = await self._redis.get(neg_key)
            if result is None:
                return None
            if isinstance(result, str):
                return json.loads(result)
            return result
        except Exception:
            return None
