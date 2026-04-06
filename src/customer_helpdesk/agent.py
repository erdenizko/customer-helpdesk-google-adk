from google.adk.agents import Agent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm
from .agents.classifier import agent as classifier_agent
from .agents.history_check import agent as history_agent
from .agents.response_enhancer import agent as enhancer_agent

# Parallel execution of Classifier + History Check
parallel_lookup = ParallelAgent(
    name="ParallelLookup",
    sub_agents=[
        classifier_agent.root_agent,
        history_agent.root_agent,
    ],
)

# Root agent orchestrates: parallel lookup then enhance
root_agent = Agent(
    name="HelpdeskOrchestrator",
    description="Main helpdesk orchestration agent",
    sub_agents=[parallel_lookup, enhancer_agent.root_agent],
    instruction="""Orchestrate the helpdesk pipeline:
    1. First, run ParallelLookup (Classifier + History Check) in parallel
    2. Then, run ResponseEnhancer with the context from both
    
    The agents will communicate via session.state:
    - classifier_intent: from Classifier
    - history_context: from HistoryCheck
    - final_response: from ResponseEnhancer (this is the final output)
    """,
)

__all__ = ["root_agent"]
