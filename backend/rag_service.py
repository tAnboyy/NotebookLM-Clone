"""RAG chat service - retrieve chunks, call LLM, persist messages."""

import os
import re

from openai import OpenAI

from backend.chat_service import save_message, load_chat
from backend.retrieval_service import retrieve_chunks

MAX_HISTORY_MESSAGES = 20
# Together AI - you have recent usage. Or :groq for Groq.
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct:together"
TOP_K = 5

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        token = os.getenv("HF_TOKEN")
        _client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )
    return _client


def _validate_citations(text: str, num_chunks: int) -> str:
    """Strip or fix citation numbers [N] where N > num_chunks."""
    if num_chunks <= 0:
        return text

    def replace_citation(match):
        n = int(match.group(1))
        if 1 <= n <= num_chunks:
            return match.group(0)
        return ""

    return re.sub(r"\[(\d+)\]", replace_citation, text)


def rag_chat(notebook_id: str, query: str, chat_history: list) -> tuple[str, list]:
    """
    RAG chat: retrieve chunks, build prompt, call LLM, persist, return answer and updated history.

    chat_history: list of [user_msg, assistant_msg] pairs (Gradio Chatbot format).
    Returns: (assistant_reply, updated_history).
    """
    save_message(notebook_id, "user", query)

    chunks = retrieve_chunks(notebook_id, query, top_k=TOP_K)

    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(f"[{i}] {c['content']}")
    context = "\n\n".join(context_parts) if context_parts else "(No relevant sources found.)"

    system_content = (
        "You are a helpful assistant. Answer ONLY from the provided context. "
        "Cite sources using [1], [2], etc. corresponding to the numbered passages. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"Context:\n{context}"
    )

    # Truncate history to last MAX_HISTORY_MESSAGES (pairs -> 2*N messages)
    max_pairs = MAX_HISTORY_MESSAGES // 2
    truncated = chat_history[-max_pairs:] if len(chat_history) > max_pairs else chat_history

    messages = [{"role": "system", "content": system_content}]
    for user_msg, asst_msg in truncated:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if asst_msg:
            messages.append({"role": "assistant", "content": asst_msg})
    messages.append({"role": "user", "content": query})

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=512,
        )
        raw_answer = response.choices[0].message.content or ""
        answer = _validate_citations(raw_answer, len(chunks))
    except Exception as e:
        answer = f"Error calling model: {e}"

    save_message(notebook_id, "assistant", answer)

    updated_history = chat_history + [[query, answer]]
    return answer, updated_history
