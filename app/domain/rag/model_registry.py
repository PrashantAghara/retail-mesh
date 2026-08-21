from dataclasses import dataclass

import psycopg2
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sentence_transformers import CrossEncoder

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, ModelLoadError
from app.core.models import SharedModels, SupportChain
from app.domain.rag.constants import SUPPORT_SYSTEM_PROMPT, simple_tokenize


@dataclass
class RAGModelsContainer:
    embedder: HuggingFaceEmbeddings
    vectorstore: PGVector
    ensemble_retriever: EnsembleRetriever
    reranker: CrossEncoder
    support_chain: SupportChain


def _fetch_all_documents(settings: Settings) -> list[Document]:
    """Fetch all documents from pgvector collection for BM25 index.

    Args:
        settings: Application settings.

    Returns:
        List of LangChain Documents.

    Raises:
        RuntimeError: If no documents found in collection.
    """
    raw_url = settings.database_url.replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(raw_url)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
        """,
        (settings.rag_collection_name,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise RuntimeError(
            f"No documents found in pgvector collection '{settings.rag_collection_name}'. "
            "Run the ingestion script first: python scripts/ingest_knowledge_base.py"
        )

    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in rows
    ]


def load_rag_models(settings: Settings, shared: SharedModels) -> RAGModelsContainer:
    """Load and initialize RAG-specific models, reusing shared embedder/LLM.

    Args:
        settings: Application settings containing model configurations.
        shared: Pre-loaded models shared across domains (embedder, LLM).

    Returns:
        RAGModelsContainer with initialized models.

    Raises:
        ConfigurationError: If required API keys are missing.
        ModelLoadError: If model loading fails.
    """
    if not settings.groq_api_key:
        raise ConfigurationError("GROQ_API_KEY is required")

    try:
        database_url = settings.database_url.replace(
            "postgresql://", "postgresql+psycopg2://"
        )
        vectorstore = PGVector(
            embeddings=shared.embedder,
            collection_name=settings.rag_collection_name,
            connection=database_url,
        )

        documents = _fetch_all_documents(settings)

        bm25_retriever = BM25Retriever.from_documents(
            documents, preprocess_func=simple_tokenize
        )
        bm25_retriever.k = settings.rag_bm25_k

        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.rag_vector_k}
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=list(settings.rag_ensemble_weights),
        )

        reranker = CrossEncoder(settings.rag_reranker_model)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SUPPORT_SYSTEM_PROMPT),
                ("human", "{query}"),
            ]
        )
        support_chain = prompt | shared.llm

        return RAGModelsContainer(
            embedder=shared.embedder,
            vectorstore=vectorstore,
            ensemble_retriever=ensemble_retriever,
            reranker=reranker,
            support_chain=support_chain,
        )
    except Exception as e:
        raise ModelLoadError(f"Failed to load RAG models: {e}") from e
