import os
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

from app.core.config import Settings
from app.nlp.constant import INTENT_EXAMPLES, INTENT_SYSTEM_PROMPT
from app.nlp.models import IntentOutput


@dataclass
class NLPModels:
    embedder: HuggingFaceEmbeddings
    category_embeddings: dict
    sentiment_analyzer: object
    intent_chain: object


def load_nlp_models(settings: Settings) -> NLPModels:
    os.environ["HF_TOKEN"] = settings.hf_token
    embedder = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
    )

    category_embeddings = {
        category: [embedder.embed_query(ex) for ex in examples]
        for category, examples in INTENT_EXAMPLES.items()
    }

    sentiment_analyzer = pipeline("sentiment-analysis", model=settings.sentiment_model)

    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.intent_llm_model)
    structured_llm = llm.with_structured_output(IntentOutput)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INTENT_SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    intent_chain = prompt | structured_llm

    return NLPModels(
        embedder=embedder,
        category_embeddings=category_embeddings,
        sentiment_analyzer=sentiment_analyzer,
        intent_chain=intent_chain,
    )
