"""Query result caching with thundering herd prevention."""

import hashlib
import unicodedata
from typing import Any, Optional

from .cache_service import CacheService

# Lazy singleton instances for standalone function access
_cache_service_instance: Optional[CacheService] = None
_query_cache_instance: Optional["QueryCache"] = None


def _get_cache_service() -> CacheService:
    """Get or create the singleton CacheService instance."""
    global _cache_service_instance
    if _cache_service_instance is None:
        from ..config import get_settings

        settings = get_settings()
        _cache_service_instance = CacheService(
            redis_url=settings.upstash_redis_rest_url,
            redis_token=settings.upstash_redis_rest_token,
        )
    return _cache_service_instance


def _get_query_cache() -> "QueryCache":
    """Get or create the singleton QueryCache instance."""
    global _query_cache_instance
    if _query_cache_instance is None:
        _query_cache_instance = QueryCache(_get_cache_service())
    return _query_cache_instance


async def get_query_result(category: str, query: str) -> Optional[dict]:
    """Standalone function to get cached query result."""
    return await _get_query_cache().get_query_result(category, query)


async def set_query_result(category: str, query: str, result: dict) -> bool:
    """Standalone function to cache query result."""
    return await _get_query_cache().set_query_result(category, query, result)


class QueryCache:
    """
    Cache for RAG results and similar ticket query results.

    Uses SHA256 hash of category + normalized query as cache key.
    TTL is 5 minutes (300 seconds) to balance freshness and load.

    Args:
        cache_service: CacheService instance for storage
    """

    KEY_TYPE = "query"
    TTL = 300  # 5 minutes

    def __init__(self, cache_service: CacheService):
        self._cache = cache_service

    def _normalize(self, text: str) -> str:
        """Normalize text using NFKC normalization, strip, and lowercase."""
        return unicodedata.normalize("NFKC", text.strip().lower())

    def _make_key(self, category: str, query: str) -> str:
        """
        Build cache key from category and normalized query.

        Key format: sha256(category + ":" + normalized_query)
        """
        normalized = self._normalize(query)
        raw_key = f"{category}:{normalized}"
        return hashlib.sha256(raw_key.encode()).hexdigest()

    async def get_query_result(self, category: str, query: str) -> Optional[dict]:
        """
        Get cached query result if available.

        Args:
            category: Query category (e.g., "technical", "billing")
            query: Raw query string

        Returns:
            Cached result dict if found, None otherwise
        """
        key = self._make_key(category, query)
        return await self._cache.get(self.KEY_TYPE, key)

    async def set_query_result(self, category: str, query: str, result: dict) -> bool:
        """
        Cache query result with TTL.

        Args:
            category: Query category
            query: Raw query string
            result: Result dict to cache (RAG results or similar ticket results)

        Returns:
            True if cached successfully, False otherwise
        """
        key = self._make_key(category, query)
        return await self._cache.set(self.KEY_TYPE, key, result, self.TTL)

    async def get_or_set(
        self,
        category: str,
        query: str,
        factory: callable,
    ) -> Optional[dict]:
        """
        Get cached result or compute and cache (thundering herd prevention).

        Uses CacheService.get_or_set which acquires a distributed lock
        to prevent multiple requests from computing the same key simultaneously.

        Args:
            category: Query category
            query: Raw query string
            factory: Async callable that computes the result if not cached

        Returns:
            Cached or computed result dict, or None on error
        """
        key = self._make_key(category, query)
        return await self._cache.get_or_set(self.KEY_TYPE, key, factory, self.TTL)
