"""Embedding service using LiteLLM for vector generation."""

import unicodedata
from typing import List

import litellm


async def generate_embedding(
    text: str, model: str = "text-embedding-ada-002"
) -> List[float]:
    normalized = unicodedata.normalize("NFKC", text.strip().lower())
    response = await litellm.aembedding(model=model, input=[normalized])
    return response["data"][0]["embedding"]
