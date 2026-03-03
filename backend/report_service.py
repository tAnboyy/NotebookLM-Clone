"""Report generation helper that prompts the LLM with retrieved chunks."""

from typing import Final

from backend.llm_client import DEFAULT_MODEL, get_llm_client
from backend.retrieval_service import retrieve_chunks

MAX_CONTEXT_ITEMS: Final[int] = 8
REPORT_SCOPE_DESCRIPTIONS = {
    "all": "PDFs, websites, and uploaded text",
    "pdf": "PDF uploads",
    "url": "ingested websites",
    "text": "uploaded text files",
}


def _is_pdf_metadata(metadata: dict) -> bool:
    name = (metadata.get("file_name") or "").lower()
    path = (metadata.get("file_path") or "").lower()
    if name.endswith(".pdf") or path.endswith(".pdf"):
        return True
    return False


def _is_text_metadata(metadata: dict) -> bool:
    name = (metadata.get("file_name") or "").lower()
    return name.endswith(".txt")


def _matches_scope(chunk: dict, scope: str) -> bool:
    if scope == "all":
        return True
    metadata = chunk.get("metadata") or {}
    if scope == "pdf":
        return _is_pdf_metadata(metadata)
    if scope == "text":
        return _is_text_metadata(metadata)
    if scope == "url":
        return bool(metadata.get("url"))
    return True


def generate_report(notebook_id: str, scope: str = "all") -> str:
    """Return a report string using the provided notebook sources."""
    normalized_scope = scope if scope in REPORT_SCOPE_DESCRIPTIONS else "all"
    desc = REPORT_SCOPE_DESCRIPTIONS[normalized_scope]
    chunks = retrieve_chunks(notebook_id, f"report summary for {desc}", top_k=16)
    filtered = [chunk for chunk in chunks if _matches_scope(chunk, normalized_scope)]
    if not filtered:
        raise ValueError(f"No {desc} chunks are available yet.")

    context_items = filtered[:MAX_CONTEXT_ITEMS]
    context_lines = [f"[{idx}] {item['content']}" for idx, item in enumerate(context_items, start=1)]
    context_block = "\n\n".join(context_lines)

    system_content = (
        f"You are NotebookLM's assistant. Using the numbered context items below that come from {desc},"
        "write a concise report that sticks to the facts."
        " Cite every referenced statement with the matching [n]."
        f"\n\nContext:\n{context_block}"
    )

    user_message = (
        "Create a report with at least the sections Summary, Key Findings, and Recommendations."
        " Bullet points are fine."
    )

    client = get_llm_client()
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ],
        max_tokens=700,
    )

    report_text = response.choices[0].message.content.strip() if response.choices else ""
    if not report_text:
        raise ValueError("The report generator returned an empty response.")
    return report_text
