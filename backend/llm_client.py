"""Helper for shared LLM-based features."""

import os

from openai import OpenAI

DEFAULT_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct:together")

_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    """Return a cached OpenAI client configured with the Hugging Face router."""
    global _client
    if _client is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN environment variable is required for LLM calls.")
        _client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )
    return _client
