from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from ...config import get_settings

settings = get_settings()

CLASSIFIER_INSTRUCTION = """You are a customer query classifier.

Analyze the user's message and classify it into ONE of these categories:
- BILLING: Subscription, payment, refund, invoice, charges questions
- TECHNICAL: API errors, integration issues, bugs, code problems, debugging
- GENERAL: Store hours, policies, basic information, greetings

Output format: JSON with single key "intent" and value being one of [billing, technical, general]

Examples:
- "cancel my subscription" -> {"intent": "billing"}
- "API returns 500 error" -> {"intent": "technical"}
- "what time do you close?" -> {"intent": "general"}
"""

root_agent = LlmAgent(
    name="Classifier",
    model=LiteLlm(model=settings.basic_model),
    description="Classifies customer queries into intent categories",
    instruction=CLASSIFIER_INSTRUCTION,
    output_key="classifier_intent",
)
