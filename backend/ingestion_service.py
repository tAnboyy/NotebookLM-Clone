"""PDF ingestion for RAG: extract text, chunk, and persist to chunks table."""

from pathlib import Path

from pypdf import PdfReader

from backend.db import supabase

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def _chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        start += step

    return chunks


def ingest_pdf_chunks(notebook_id: str, source_id: str, pdf_path: Path) -> int:
    """Extract and store chunks for a single PDF. Returns number of chunks inserted."""
    text = _extract_pdf_text(pdf_path)
    chunks = _chunk_text(text)

    supabase.table("chunks").delete().eq("notebook_id", notebook_id).eq("source_id", source_id).execute()

    if not chunks:
        return 0

    rows = [
        {
            "notebook_id": notebook_id,
            "source_id": source_id,
            "content": chunk,
            "metadata": {
                "file_name": source_id,
                "file_path": str(pdf_path),
                "chunk_index": index,
                "total_chunks": len(chunks),
            },
        }
        for index, chunk in enumerate(chunks)
    ]

    batch_size = 100
    for offset in range(0, len(rows), batch_size):
        supabase.table("chunks").insert(rows[offset:offset + batch_size]).execute()

    return len(rows)


def remove_chunks_for_source(notebook_id: str, source_id: str) -> None:
    """Delete all chunks tied to one source file for a notebook."""
    supabase.table("chunks").delete().eq("notebook_id", notebook_id).eq("source_id", source_id).execute()
