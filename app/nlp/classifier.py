import numpy as np

from app.nlp.model_loaders import NLPModels


def classify_intent(query: str, models: NLPModels):
    query_vec = np.array(models.embedder.embed_query(query))
    scores = {}
    for category, vecs in models.category_embeddings.items():
        sims = [
            np.dot(query_vec, np.array(v))
            / (np.linalg.norm(query_vec) * np.linalg.norm(v))
            for v in vecs
        ]
        scores[category] = max(sims)
    best = max(scores, key=scores.get)
    return best, scores[best], scores
