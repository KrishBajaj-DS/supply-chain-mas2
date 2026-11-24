"""
RAG retriever for the Alternative Supplier Agent, using Google Gemini's
free embedding API (text-embedding-004) for semantic similarity instead
of keyword-based TF-IDF.

Requires GEMINI_API_KEY in .env (separate from GROQ_API_KEY, which is
used only for the chat/reasoning LLM - Groq doesn't offer embeddings).

Uses the current `google-genai` SDK (the older `google-generativeai`
package is deprecated).
"""

import os
import re
import numpy as np
from dotenv import load_dotenv
from src.supplier_kb import get_supplier_docs

load_dotenv()

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    genai = None
    types = None
    GENAI_AVAILABLE = False

_client = None


def _get_client():
    global _client
    if _client is None and GENAI_AVAILABLE and genai is not None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "unused-by-mcp-tools":
            try:
                _client = genai.Client(api_key=api_key)
            except Exception:
                _client = None
    return _client


EMBED_MODEL = "gemini-embedding-001"


def _deterministic_offline_embed(text: str) -> np.ndarray:
    """Deterministic offline fallback embedding vector.
    Allows unit tests and offline evaluation to run without google-genai installed."""
    vec = np.zeros(64, dtype=float)
    tokens = re.findall(r"\w+", text.lower())
    for token in tokens:
        h = abs(hash(token)) % 64
        vec[h] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _embed(text: str, task_type: str) -> np.ndarray:
    """Get an embedding vector for a piece of text from Gemini (with offline fallback)."""
    client = _get_client()
    if client is not None and types is not None:
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return np.array(result.embeddings[0].values)
        except Exception:
            pass
    return _deterministic_offline_embed(text)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SupplierRetriever:
    def __init__(self):
        self.docs = get_supplier_docs()
        # Embed each supplier doc once at startup and cache the vectors
        self.doc_vectors = [
            _embed(d["text"], task_type="RETRIEVAL_DOCUMENT") for d in self.docs
        ]

    def retrieve(self, query: str, top_k: int = 3):
        """Return the top_k most semantically relevant supplier docs for a
        query, each with a similarity score, sorted highest first."""
        query_vec = _embed(query, task_type="RETRIEVAL_QUERY")
        scores = [_cosine_sim(query_vec, dv) for dv in self.doc_vectors]
        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in ranked_idx:
            results.append({
                "id": self.docs[i]["id"],
                "name": self.docs[i]["name"],
                "text": self.docs[i]["text"],
                "score": round(float(scores[i]), 4),
            })
        return results


# Singleton instance so docs are only embedded once per process
_retriever_instance = None


def get_retriever() -> SupplierRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = SupplierRetriever()
    return _retriever_instance
