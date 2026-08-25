from dataclasses import dataclass

from langchain.agents import create_agent

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, ModelLoadError
from app.core.models import FulfillmentAgent, SharedModels
from app.domain.vision.constants import VISION_SYSTEM_PROMPT
from app.domain.vision.service import load_shelf_model
from app.domain.vision.tools import build_vision_tools


@dataclass
class VisionModelsContainer:
    agent: FulfillmentAgent


def load_vision_models(
    settings: Settings, shared: SharedModels
) -> VisionModelsContainer:
    if not settings.vision_weights_path:
        raise ConfigurationError("VISION_WEIGHTS_PATH is required")

    try:
        shelf_model = load_shelf_model(settings.vision_weights_path)
        tools = build_vision_tools(shelf_model, shared)
        agent = create_agent(
            model=shared.llm, tools=tools, system_prompt=VISION_SYSTEM_PROMPT
        )
        return VisionModelsContainer(agent=agent)
    except Exception as e:
        raise ModelLoadError(f"Failed to load vision models: {e}") from e
