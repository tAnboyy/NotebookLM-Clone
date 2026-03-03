"""Retrieval service - vector similarity search for RAG."""

from backend.db import supabase
from backend.embedding_service import encode


def retrieve_chunks(notebook_id: str, query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve top-k chunks for a query, filtered by notebook_id.

    Returns list of dicts with keys: id, content, metadata, similarity.
    """
    if not query or not query.strip():
        return []

    query_embedding = encode([query.strip()], task="search_query")[0]

    try:
        result = supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "p_notebook_id": notebook_id,
            },
        ).execute()

        rows = result.data or []
        return [
            {
                "id": str(r["id"]),
                "content": r["content"],
                "metadata": r.get("metadata") or {},
                "similarity": float(r.get("similarity", 0)),
            }
            for r in rows
        ]
    except Exception:
        return []
