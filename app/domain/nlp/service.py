from app.core.config import Settings
from app.domain.nlp.classifier import classify_intent
from app.domain.nlp.model_registry import NLPModelsContainer
from app.domain.nlp.schemas import AnalyzeResponse
from app.domain.nlp.sentiment import get_sentiment


def get_intent(query: str, models: NLPModelsContainer, settings: Settings) -> AnalyzeResponse:
    """Determine intent and sentiment for a query.
    
    Uses embedding-based classification first, falls back to LLM if confidence is low.
    
    Args:
        query: User query to analyze.
        models: Loaded NLP models container.
        settings: Application settings.
        
    Returns:
        AnalyzeResponse with category, confidence, sentiment, and method used.
    """
    category, confidence, _ = classify_intent(query, models)
    sentiment = get_sentiment(query, models.sentiment_analyzer)

    if confidence >= settings.intent_confidence_threshold:
        return AnalyzeResponse(
            query=query,
            category=category,
            confidence=round(float(confidence), 3),
            sentiment=sentiment,
            method="embedding",
        )

    result = models.intent_chain.invoke({"query": query})
    return AnalyzeResponse(
        query=query,
        category=result.category,
        confidence=None,
        sentiment=result.sentiment,
        method="llm_fallback",
        reasoning=result.reasoning,
    )
