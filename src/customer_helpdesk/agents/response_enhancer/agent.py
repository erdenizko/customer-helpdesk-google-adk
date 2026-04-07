from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from ...config import get_settings
from ...services.vector_store import get_vector_store
from ...services.embedding_cache import get_embedding
from ...services.query_cache import QueryCache
from ...services.cache_service import CacheService

settings = get_settings()
_cache_service = CacheService(
    redis_url=settings.upstash_redis_rest_url,
    redis_token=settings.upstash_redis_rest_token,
)
_query_cache = QueryCache(_cache_service)


async def rag_retrieve(query: str, category: str, limit: int = 3) -> dict:
    """Retrieve relevant documents from Qdrant for RAG.

    Only performs RAG for technical queries. Returns empty results for other categories.
    """
    # Only search for technical queries
    if category != "technical":
        return {
            "status": "success",
            "results": [],
            "skipped": True,
            "reason": "RAG only available for technical queries",
        }

    cached_result = await _query_cache.get_query_result(category, query)
    if cached_result is not None:
        return {
            "status": "success",
            "results": cached_result["results"],
            "skipped": False,
        }

    vs = await get_vector_store()
    query_vector = await get_embedding(query, _cache_service)
    results = await vs.search(
        query_vector=query_vector,
        filter_conditions={"category": {"$eq": category}},
        limit=limit,
    )

    await _query_cache.set_query_result(category, query, {"results": results})

    return {"status": "success", "results": results, "skipped": False}


RAG_TOOLS = [
    FunctionTool(func=rag_retrieve),
]

ENHANCER_INSTRUCTION = """You are a response enhancement agent.

Context available in session state:
- classifier_intent: The classified intent from Classifier agent (billing/technical/general)
- history_context: User history and similar issues from HistoryCheck agent
- user_query: The original user message

Task:
1. If classifier_intent == "technical": Use rag_retrieve tool to get relevant KB docs
2. Use history_context to personalize response based on user's history and similar past issues
3. Generate a helpful, concise, and accurate response

Important:
- RAG tool is only useful for technical queries
- Always consider the user's history when formulating a response
- For billing queries, do NOT use rag_retrieve
- For general queries, provide friendly, helpful responses

Output format: JSON with "response" key containing the response text.
"""

root_agent = LlmAgent(
    name="ResponseEnhancer",
    model=LiteLlm(model=settings.complex_model),
    description="Enhances responses using RAG and history",
    instruction=ENHANCER_INSTRUCTION,
    tools=RAG_TOOLS,
    output_key="final_response",
)
