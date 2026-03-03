"""Shared embedding service - 384-dim vectors for RAG (ingestion + retrieval). Uses MiniLM for low memory."""

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(texts: list[str], task: str = "search_document") -> list[list[float]]:
    """
    Embed texts. Returns list of 384-dim vectors.

    Args:
        texts: List of strings to embed.
        task: Unused (MiniLM doesn't need prefix); kept for API compatibility.
    """
    if not texts:
        return []

    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [e.tolist() for e in embeddings]
