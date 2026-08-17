from app.core.config import Settings
from app.nlp.classifier import classify_intent
from app.nlp.model_loaders import NLPModels
from app.nlp.sentiment import get_sentiment


def get_intent(query: str, models: NLPModels, settings: Settings) -> dict:
    category, confidence, _ = classify_intent(query, models)
    sentiment = get_sentiment(query, models.sentiment_analyzer)

    if confidence >= settings.intent_confidence_threshold:
        return {
            "query": query,
            "category": category,
            "confidence": round(float(confidence), 3),
            "sentiment": sentiment,
            "method": "embedding",
        }

    result = models.intent_chain.invoke({"query": query})
    return {
        "query": query,
        "category": result.category,
        "confidence": None,
        "sentiment": result.sentiment,
        "method": "llm_fallback",
        "reasoning": result.reasoning,
    }
