from dataclasses import dataclass

from groq import Groq

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, ModelLoadError


@dataclass
class VoiceModelsContainer:
    groq_client: Groq


def load_voice_models(settings: Settings) -> VoiceModelsContainer:
    if not settings.groq_api_key:
        raise ConfigurationError("GROQ_API_KEY is required")
    try:
        groq_client = Groq(api_key=settings.groq_api_key)
        return VoiceModelsContainer(groq_client=groq_client)
    except Exception as e:
        raise ModelLoadError(f"Failed to load voice models: {e}") from e
