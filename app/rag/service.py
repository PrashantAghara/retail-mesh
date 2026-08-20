from app.rag.model_registry import RAGModels


def retrieve(query: str, models: RAGModels, top_k: int = 3):
    candidates = models.ensemble_retriever.invoke(query)
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = models.reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def support_agent(query: str, models: RAGModels) -> dict:
    docs = retrieve(query, models, top_k=3)

    context = "\n\n".join(
        f"[{doc.metadata.get('doc_type')}] {doc.page_content}" for doc in docs
    )

    response = models.support_chain.invoke({"context": context, "query": query})

    return {
        "query": query,
        "answer": response.content,
        "sources": [
            {
                "title": doc.metadata.get("title"),
                "category": doc.metadata.get("category"),
            }
            for doc in docs
        ],
    }
