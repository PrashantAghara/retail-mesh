from dataclasses import dataclass

from langchain.agents import create_agent

from app.core.config import Settings
from app.core.database import get_engine
from app.core.exceptions import ConfigurationError, ModelLoadError
from app.core.models import FulfillmentAgent, SharedModels
from app.domain.fulfillment.tools import build_fulfillment_tools

FULFILLMENT_SYSTEM_PROMPT = """You are RetailMesh's fulfillment assistant. You help customers check product
availability, place orders, track existing orders, and cancel orders. Always use the
provided tools to get real data — never guess stock levels or order statuses. Be concise
and friendly in your responses."""


@dataclass
class FulfillmentModelsContainer:
    agent: FulfillmentAgent


def load_fulfillment_models(
    settings: Settings, shared: SharedModels
) -> FulfillmentModelsContainer:
    """Load and initialize fulfillment agent, reusing the shared LLM.

    Args:
        settings: Application settings containing model configurations.
        shared: Pre-loaded models shared across domains (embedder, LLM).

    Returns:
        FulfillmentModelsContainer with initialized agent.

    Raises:
        ConfigurationError: If required API keys are missing.
        ModelLoadError: If agent creation fails.
    """
    if not settings.groq_api_key:
        raise ConfigurationError("GROQ_API_KEY is required")

    try:
        engine = get_engine()
        tools = build_fulfillment_tools(engine, settings)

        agent = create_agent(
            model=shared.llm, tools=tools, system_prompt=FULFILLMENT_SYSTEM_PROMPT
        )
        return FulfillmentModelsContainer(agent=agent)
    except Exception as e:
        raise ModelLoadError(f"Failed to load fulfillment models: {e}") from e
