from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from ...config import get_settings
from ...services.database import get_user_tickets, search_similar_tickets

settings = get_settings()


async def lookup_user_history(user_id: str, limit: int = 5) -> dict:
    """Look up recent tickets for this user"""
    tickets = await get_user_tickets(user_id, limit)
    return {
        "status": "success",
        "tickets": [
            {
                "id": t.id,
                "category": t.category.value,
                "subject": t.subject,
                "status": t.status.value,
                "created_at": t.created_at.isoformat(),
            }
            for t in tickets
        ],
    }


async def lookup_similar_issues(query: str, category: str, limit: int = 3) -> dict:
    """Search for similar resolved tickets"""
    tickets = await search_similar_tickets(query, category, limit)
    return {
        "status": "success",
        "similar_tickets": [
            {
                "id": t.id,
                "subject": t.subject,
                "resolution": t.resolution,
                "category": t.category.value,
            }
            for t in tickets
        ],
    }


HISTORY_TOOLS = [
    FunctionTool(func=lookup_user_history),
    FunctionTool(func=lookup_similar_issues),
]

HISTORY_INSTRUCTION = """You are a history lookup agent.

Use the provided tools to gather context:
1. lookup_user_history - Get recent tickets for the user from session.user_id
2. lookup_similar_issues - Search for similar resolved tickets by category

Return a JSON summary with both user history and similar issues.
"""

root_agent = LlmAgent(
    name="HistoryCheck",
    model=LiteLlm(model=settings.basic_model),
    description="Checks user history and similar resolved tickets",
    instruction=HISTORY_INSTRUCTION,
    tools=HISTORY_TOOLS,
    output_key="history_context",  # UNIQUE key for parallel execution
)
