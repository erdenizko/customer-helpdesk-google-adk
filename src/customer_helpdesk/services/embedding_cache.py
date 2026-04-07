"""Embedding cache service using CacheService for storage."""

import hashlib
import unicodedata
from typing import List, Optional

from .cache_service import CacheService
from .embedding_service import generate_embedding

TTL_SECONDS = 3600
NEGATIVE_CACHE_TTL = 30
KEY_TYPE = "emb"
ERROR_MARKER = "error"


def _normalize_text(text: str) -> str:
    """Normalize text using NFKC, strip, and lowercase."""
    return unicodedata.normalize("NFKC", text.strip().lower())


def _compute_cache_key(text: str, document_id: Optional[str] = None) -> str:
    """Compute SHA256 hash of normalized text for cache key."""
    normalized = _normalize_text(text)
    key_input = normalized if not document_id else f"{document_id}:{normalized}"
    return hashlib.sha256(key_input.encode("utf-8")).hexdigest()


async def get_embedding(
    text: str,
    cache: CacheService,
    model: str = "text-embedding-ada-002",
    document_id: Optional[str] = None,
) -> List[float]:
    """Get embedding for text, checking cache first.

    Supports negative caching: failed LiteLLM calls are cached for 30s
    to prevent thundering herd on service outages.
    """
    cache_key = _compute_cache_key(text, document_id)

    cached = await cache.get(KEY_TYPE, cache_key)
    if cached is not None:
        if cached.get(ERROR_MARKER):
            raise Exception(f"Embedding error (cached): {cached[ERROR_MARKER]}")
        return cached["embedding"]

    try:
        embedding = await generate_embedding(text, model)
        await cache.set(KEY_TYPE, cache_key, {"embedding": embedding}, TTL_SECONDS)
        return embedding
    except Exception as e:
        await cache.set(KEY_TYPE, cache_key, {ERROR_MARKER: str(e)}, NEGATIVE_CACHE_TTL)
        raise


async def invalidate_embedding(
    text: str, cache: CacheService, document_id: Optional[str] = None
) -> bool:
    """Invalidate cached embedding for text."""
    cache_key = _compute_cache_key(text, document_id)
    return await cache.delete(KEY_TYPE, cache_key)
