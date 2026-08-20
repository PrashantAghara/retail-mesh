import os
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, ModelLoadError
from app.core.models import IntentChain, SentimentAnalyzer
from app.domain.nlp.constants import INTENT_EXAMPLES, INTENT_SYSTEM_PROMPT
from app.domain.nlp.schemas import IntentOutput


@dataclass
class NLPModelsContainer:
    embedder: HuggingFaceEmbeddings
    category_embeddings: dict[str, list[list[float]]]
    sentiment_analyzer: SentimentAnalyzer
    intent_chain: IntentChain


def load_nlp_models(settings: Settings) -> NLPModelsContainer:
    """Load and initialize all NLP models.
    
    Args:
        settings: Application settings containing model configurations.
        
    Returns:
        NLPModelsContainer with initialized models.
        
    Raises:
        ConfigurationError: If required API keys are missing.
        ModelLoadError: If model loading fails.
    """
    if not settings.groq_api_key:
        raise ConfigurationError("GROQ_API_KEY is required")
    
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
    
    try:
        embedder = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        category_embeddings = {
            category: [embedder.embed_query(ex) for ex in examples]
            for category, examples in INTENT_EXAMPLES.items()
        }

        sentiment_analyzer = pipeline("sentiment-analysis", model=settings.sentiment_model)

        llm = ChatGroq(api_key=settings.groq_api_key, model=settings.intent_llm_model)
        structured_llm = llm.with_structured_output(IntentOutput)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INTENT_SYSTEM_PROMPT),
                ("human", "{query}"),
            ]
        )
        intent_chain = prompt | structured_llm

        return NLPModelsContainer(
            embedder=embedder,
            category_embeddings=category_embeddings,
            sentiment_analyzer=sentiment_analyzer,
            intent_chain=intent_chain,
        )
    except Exception as e:
        raise ModelLoadError(f"Failed to load NLP models: {e}") from e
