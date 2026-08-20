from langchain_core.documents import Document

from app.core.config import get_settings
from app.domain.rag.model_registry import RAGModelsContainer
from app.domain.rag.schemas import Source, SupportQueryResponse


def retrieve(
    query: str, models: RAGModelsContainer, top_k: int | None = None
) -> list[Document]:
    """Retrieve relevant documents for a query.
    
    Args:
        query: Search query.
        models: Loaded RAG models container.
        top_k: Number of documents to return (defaults to settings.rag_top_k).
        
    Returns:
        List of retrieved documents.
    """
    if top_k is None:
        top_k = get_settings().rag_top_k

    candidates = models.ensemble_retriever.invoke(query)
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = models.reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def support_agent(query: str, models: RAGModelsContainer) -> SupportQueryResponse:
    """Generate support response using RAG.
    
    Args:
        query: User query.
        models: Loaded RAG models container.
        
    Returns:
        SupportQueryResponse with answer and sources.
    """
    top_k = get_settings().rag_top_k
    docs = retrieve(query, models, top_k=top_k)

    context = "\n\n".join(
        f"[{doc.metadata.get('doc_type')}] {doc.page_content}" for doc in docs
    )

    response = models.support_chain.invoke({"context": context, "query": query})

    return SupportQueryResponse(
        query=query,
        answer=response.content,
        sources=[
            Source(
                title=doc.metadata.get("title"),
                category=doc.metadata.get("category"),
            )
            for doc in docs
        ],
    )
