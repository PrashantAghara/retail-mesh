from app.nlp.constant import SENTIMENT_LABEL_MAP


def get_sentiment(query: str, sentiment_analyzer) -> str:
    result = sentiment_analyzer(query)[0]
    label = result["label"]
    return SENTIMENT_LABEL_MAP.get(label, label.lower())
