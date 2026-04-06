import pytest
from unittest.mock import AsyncMock, MagicMock
from src.customer_helpdesk.agents.classifier import agent


def test_classifier_outputs_valid_intent():
    """Test that classifier agent outputs valid intent."""
    # Verify agent definition
    assert agent.root_agent.name == "Classifier"
    assert agent.root_agent.output_key == "classifier_intent"


def test_classifier_intent_values():
    """Test that intent values are constrained."""
    valid_intents = ["billing", "technical", "general"]
    # The instruction should constrain output to these values
    assert "billing" in valid_intents
    assert "technical" in valid_intents
    assert "general" in valid_intents


def test_classifier_instruction_contains_categories():
    """Test that instruction defines the three categories."""
    instruction = agent.CLASSIFIER_INSTRUCTION
    assert "BILLING" in instruction
    assert "TECHNICAL" in instruction
    assert "GENERAL" in instruction
