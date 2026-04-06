from qdrant_client import AsyncQdrantClient, models
from typing import Optional
from ..config import get_settings

settings = get_settings()


class VectorStoreService:
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection

    async def ensure_collection(self):
        """Create collection if not exists"""
        collections = await self.client.get_collections()
        if self.collection not in [c.name for c in collections.collections]:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=models.Distance.COSINE,
                ),
            )

    async def upsert(
        self,
        points: list[dict],  # [{"id": str, "vector": list[float], "payload": dict}]
        batch_size: int = 100,
    ):
        await self.client.upsert(
            collection_name=self.collection, points=points, wait=True
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
