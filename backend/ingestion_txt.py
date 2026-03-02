"""
Text file ingestion pipeline.
Handles .txt upload → extract → clean → save to Supabase DB + Storage.
"""

import chardet
import re
from datetime import datetime
from uuid import uuid4

from backend.db import supabase
from backend.storage import save_file, get_sources_path

import os
from sentence_transformers import SentenceTransformer

# Load model once at module level (not on every call)
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# ── Constants ────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


# ── Text Processing ──────────────────────────────────────────

def detect_encoding(file_bytes: bytes) -> str:
    """
    Detects encoding of raw bytes.
    Falls back to utf-8 if confidence is low.
    """
    result = chardet.detect(file_bytes)
    encoding = result.get("encoding") or "utf-8"
    confidence = result.get("confidence") or 0

    if confidence < 0.7:
        return "utf-8"

    return encoding


def clean_text(text: str) -> str:
    """
    Cleans raw extracted text.
    - Removes null bytes
    - Removes control characters (keeps newlines + tabs)
    - Normalizes excessive blank lines
    - Strips leading/trailing whitespace
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove control characters except \n and \t
    text = "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or ch >= " "
    )

    # Normalize 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Supabase DB Operations ───────────────────────────────────

def _create_source_record(
    source_id: str,
    notebook_id: str,
    user_id: str,
    filename: str,
    storage_path: str
) -> None:
    """Insert a new source row with PENDING status."""
    supabase.table("sources").insert({
        "id": source_id,
        "notebook_id": notebook_id,
        "user_id": user_id,
        "filename": filename,
        "file_type": "txt",
        "status": "PENDING",
        "storage_path": storage_path,
    }).execute()

# ── Chunking ─────────────────────────────────────────────────
def chunk_text(text: str, source_id: str, notebook_id: str, filename: str = "") -> list[dict]:
    words = text.split()
    chunk_size = 400
    overlap = 40
    chunks = []
    i = 0

    # Calculate total chunks upfront
    total_chunks = max(1, (len(words) + chunk_size - overlap - 1) // (chunk_size - overlap))

    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        content = " ".join(chunk_words)
        chunks.append({
            "id": str(uuid4()),
            "source_id": source_id,
            "notebook_id": notebook_id,
            "content": content,
            "chunk_index": len(chunks),
            "metadata": {
                "word_count": len(chunk_words),
                "file_name": filename,
                "chunk_index": len(chunks),
                "total_chunks": total_chunks,
            }
        })
        i += chunk_size - overlap

    return chunks


# ── Embed + Store ─────────────────────────────────────────────
def embed_and_store_chunks(chunks: list[dict]) -> None:
    """
    Embed chunks using sentence-transformers and store in pgvector.
    """
    if not chunks:
        return

    # Embed all chunks in one batch
    texts = [c["content"] for c in chunks]
    embeddings = _model.encode(texts, show_progress_bar=False)

    # Build rows for Supabase insert
    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "id": str(chunk["id"]),
            "source_id": str(chunk["source_id"]),
            "notebook_id": str(chunk["notebook_id"]),
            "content": chunk["content"],
            "embedding": embedding.tolist(),
            "metadata": chunk["metadata"]
        })

    try:
        supabase.table("chunks").insert(rows).execute()
        print(f"✅ Inserted {len(rows)} chunks into pgvector")
    except Exception as e:
        print(f"❌ Failed to insert chunks: {e}")
        raise

def _update_source_ready(
    source_id: str,
    extracted_text: str,
    metadata: dict
) -> None:
    """Mark source as READY with extracted text and metadata."""
    supabase.table("sources").update({
        "status": "READY",
        "extracted_text": extracted_text,
        "metadata": metadata,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", source_id).execute()


def _update_source_failed(source_id: str, error: str) -> None:
    """Mark source as FAILED with error message in metadata."""
    supabase.table("sources").update({
        "status": "FAILED",
        "metadata": {"error": error},
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", source_id).execute()


# ── Main Ingestion Function ──────────────────────────────────

def ingest_txt(
    file_bytes: bytes,
    filename: str,
    notebook_id: str,
    user_id: str
) -> dict:
    """
    Full pipeline for a .txt file upload:
    1. Validate size
    2. Upload raw file to Supabase Storage
    3. Create source record (PENDING)
    4. Detect encoding + decode
    5. Clean text
    6. Update source record (READY)
    7. Return result dict

    Returns dict with source_id, filename, status, metadata.
    Raises ValueError on validation errors.
    """

    # ── Validate ─────────────────────────────────────────────
    if not file_bytes:
        raise ValueError("Empty file — nothing to ingest.")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(f"File too large. Max size is 10MB.")

    if not filename.lower().endswith(".txt"):
        raise ValueError("Only .txt files are accepted here.")

    # ── Generate IDs ─────────────────────────────────────────
    source_id = str(uuid4())

    # ── Upload raw file to Supabase Storage ──────────────────
    sources_path = get_sources_path(user_id, notebook_id)
    storage_path = f"{sources_path}/{source_id}_{filename}"

    save_file(storage_path, file_bytes)

    # ── Create DB record (PENDING) ───────────────────────────
    _create_source_record(
        source_id=source_id,
        notebook_id=notebook_id,
        user_id=user_id,
        filename=filename,
        storage_path=storage_path
    )

    # ── Extract + Clean ───────────────────────────────────────
    try:
        encoding = detect_encoding(file_bytes)
        raw_text = file_bytes.decode(encoding, errors="replace")
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            raise ValueError("No text content found after cleaning.")

        metadata = {
            "encoding": encoding,
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "line_count": cleaned_text.count("\n") + 1,
            "size_bytes": len(file_bytes),
        }

        # ── Update DB record (READY) ──────────────────────────
        _update_source_ready(source_id, cleaned_text, metadata)
        
        # ── Chunk + Embed + Store ─────────────────────────────
        print(f"🔄 Starting chunking for {filename}...")
        chunks = chunk_text(cleaned_text, source_id, notebook_id, filename=filename)
        print(f"🔄 Created {len(chunks)} chunks, embedding now...")
        embed_and_store_chunks(chunks)

        return {
            "source_id": source_id,
            "filename": filename,
            "status": "READY",
            "metadata": metadata,
            "extracted_text": cleaned_text,
            "chunks_created": len(chunks),
        }

    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        _update_source_failed(source_id, str(e))
        raise


# ── List Sources for a Notebook ──────────────────────────────

def list_sources(notebook_id: str) -> list[dict]:
    """
    Returns all sources for a notebook ordered by created_at.
    """
    result = supabase.table("sources")\
        .select("id, filename, file_type, status, metadata, created_at")\
        .eq("notebook_id", notebook_id)\
        .order("created_at")\
        .execute()

    return result.data or []