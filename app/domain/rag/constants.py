import re

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "do",
    "does",
    "my",
    "your",
    "i",
    "you",
    "it",
    "this",
    "that",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "and",
    "or",
    "how",
}


def simple_tokenize(text: str):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


SUPPORT_SYSTEM_PROMPT = """You are RetailMesh's customer support assistant. Answer the customer's
question using ONLY the context provided below. If the context doesn't contain
enough information to answer confidently, say so honestly instead of guessing.

Keep answers concise and friendly. Don't mention "the context" or "the documents"
in your reply — answer as if you simply know this information.

Context:
{context}"""
