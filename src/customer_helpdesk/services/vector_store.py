import structlog
from qdrant_client import AsyncQdrantClient, models
from typing import Optional
from ..config import get_settings
from .cache_service import CacheService
from .embedding_service import generate_embedding

logger = structlog.get_logger(__name__)

settings = get_settings()

EMBEDDING_KEY_TYPE = "emb"


class VectorStoreService:
    def __init__(self, cache: Optional[CacheService] = None):
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self._cache = cache

    async def ensure_collection(self):
        """Create collection if not exists"""
        collections = await self.client.get_collections()
        if self.collection not in [c.name for c in collections.collections]:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE,
                ),
            )

    async def invalidate_document(self, document_id: str) -> bool:
        """Invalidate all embedding cache entries for a document."""
        if not self._cache:
            return False

        try:
            pattern = f"v1:cache:{EMBEDDING_KEY_TYPE}:{document_id}:*"
            cursor = 0
            deleted_count = 0

            while True:
                cursor, keys = await self._cache._redis.scan(
                    cursor=cursor, match=pattern, count=100
                )
                if keys:
                    for key in keys:
                        await self._cache._redis.delete(key)
                        deleted_count += 1
                if cursor == 0:
                    break

            logger.info(
                "document_cache_invalidated",
                document_id=document_id,
                deleted=deleted_count,
            )
            return True
        except Exception as e:
            logger.error(
                "cache_invalidation_failed", document_id=document_id, error=str(e)
            )
            return False

    async def upsert(
        self,
        points: list[dict],
        batch_size: int = 100,
    ):
        invalidated_docs = set()
        failed_point_ids = []
        valid_points = []

        for point in points:
            chunk_text = point.get("payload", {}).get("text")
            document_id = point.get("payload", {}).get("document_id")

            if document_id and document_id not in invalidated_docs:
                try:
                    existing = await self.client.retrieve(
                        collection_name=self.collection,
                        ids=[point.get("id")],
                        with_payload=True,
                    )
                    if existing:
                        await self.invalidate_document(document_id)
                        invalidated_docs.add(document_id)
                except Exception:
                    pass

            if chunk_text:
                try:
                    point["vector"] = await generate_embedding(chunk_text)
                    valid_points.append(point)
                except Exception as e:
                    logger.error(
                        "embedding_failed",
                        error=str(e),
                        point_id=point.get("id"),
                    )
                    failed_point_ids.append(point.get("id"))
            else:
                valid_points.append(point)

        if failed_point_ids:
            logger.warning(
                "upsert_skipping_failed_embeddings",
                failed_count=len(failed_point_ids),
                failed_ids=failed_point_ids,
            )

        if valid_points:
            await self.client.upsert(
                collection_name=self.collection, points=valid_points, wait=True
            )

    async def search(
        self,
        query_vector: list[float],
        filter_conditions: Optional[dict] = None,
        score_threshold: float = 0.7,
        limit: int = 5,
    ) -> list[dict]:
        results = await self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=filter_conditions,
            score_threshold=score_threshold,
            limit=limit,
            with_payload=True,
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

    async def close(self):
        await self.client.close()


vector_store = VectorStoreService()


async def get_vector_store() -> VectorStoreService:
    return vector_store
