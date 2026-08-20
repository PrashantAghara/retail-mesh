from dataclasses import dataclass

import psycopg2
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sentence_transformers import CrossEncoder

from app.core.config import Settings
from app.rag.constants import SUPPORT_SYSTEM_PROMPT, simple_tokenize

COLLECTION_NAME = "retailmesh_knowledge_base"


@dataclass
class RAGModels:
    embedder: HuggingFaceEmbeddings
    vectorstore: PGVector
    ensemble_retriever: EnsembleRetriever
    reranker: CrossEncoder
    support_chain: object


def _fetch_all_documents(settings: Settings) -> list[Document]:
    """
    Reads whatever is already seeded in the pgvector collection and
    reconstructs Document objects — used only to build BM25's in-memory
    index. Assumes the ingestion script has already populated the table.
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
        (COLLECTION_NAME,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise RuntimeError(
            f"No documents found in pgvector collection '{COLLECTION_NAME}'. "
            "Run the ingestion script first: python scripts/ingest_knowledge_base.py"
        )

    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in rows
    ]


def load_rag_models(settings: Settings) -> RAGModels:
    embedder = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    database_url = settings.database_url.replace(
        "postgresql://", "postgresql+psycopg2://"
    )
    vectorstore = PGVector(
        embeddings=embedder,
        collection_name=COLLECTION_NAME,
        connection=database_url,
    )

    documents = _fetch_all_documents(settings)

    bm25_retriever = BM25Retriever.from_documents(
        documents, preprocess_func=simple_tokenize
    )
    bm25_retriever.k = 8

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7],
    )

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.intent_llm_model)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPPORT_SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    support_chain = prompt | llm

    return RAGModels(
        embedder=embedder,
        vectorstore=vectorstore,
        ensemble_retriever=ensemble_retriever,
        reranker=reranker,
        support_chain=support_chain,
    )
