from app.core.models import SentimentAnalyzer
from app.domain.nlp.constants import SENTIMENT_LABEL_MAP


def get_sentiment(query: str, sentiment_analyzer: SentimentAnalyzer) -> str:
    """Get sentiment label for a query.
    
    Args:
        query: Text to analyze.
        sentiment_analyzer: HuggingFace sentiment analysis pipeline.
        
    Returns:
        Sentiment label: 'positive', 'negative', or 'neutral'.
    """
    result = sentiment_analyzer(query)[0]
    label = result["label"]
    return SENTIMENT_LABEL_MAP.get(label, label.lower())
