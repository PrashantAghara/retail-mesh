import numpy as np

from app.domain.nlp.model_registry import NLPModelsContainer


def classify_intent(query: str, models: NLPModelsContainer) -> tuple[str, float, dict[str, float]]:
    """Classify intent using embedding similarity.
    
    Args:
        query: User query to classify.
        models: Loaded NLP models container.
        
    Returns:
        Tuple of (best_category, best_score, all_scores).
    """
    query_vec = np.array(models.embedder.embed_query(query))
    scores: dict[str, float] = {}
    for category, vecs in models.category_embeddings.items():
        sims = [
            np.dot(query_vec, np.array(v))
            / (np.linalg.norm(query_vec) * np.linalg.norm(v))
            for v in vecs
        ]
        scores[category] = max(sims)
    best = max(scores, key=scores.get)
    return best, scores[best], scores
