"""
Quiz generation service.
Retrieves chunks from Supabase, sends to HF Inference API, saves artifact.
"""

import os
import json
import re
import requests
from backend.db import supabase
from backend.artifacts_service import create_artifact
from huggingface_hub import InferenceClient


HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Retrieval 
def _get_chunks_for_notebook(notebook_id: str, limit: int = 15) -> list[str]:
    """Fetch chunks for a notebook, spread across all sources."""
    # Get all unique source_ids first
    sources_result = (
        supabase.table("chunks")
        .select("source_id")
        .eq("notebook_id", notebook_id)
        .execute()
    )
    source_ids = list({row["source_id"] for row in (sources_result.data or [])})
    
    # Fetch a few chunks from each source
    all_chunks = []
    per_source = max(2, limit // len(source_ids)) if source_ids else limit
    for source_id in source_ids:
        result = (
            supabase.table("chunks")
            .select("content")
            .eq("notebook_id", notebook_id)
            .eq("source_id", source_id)
            .limit(per_source)
            .execute()
        )
        all_chunks += [row["content"] for row in (result.data or [])]
    
    return all_chunks[:limit]


# Prompt Builder 
def _build_prompt(context: str) -> str:
    return f"""You are a quiz generator. Based on the context below, generate a quiz with exactly 5 questions.
Use a mix of question types: multiple choice (A/B/C/D), true/false, and short answer.

Format your response as a JSON array only, no preamble:
[
  {{"type": "multiple_choice", "question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "A"}},
  {{"type": "true_false", "question": "...", "answer": "True"}},
  {{"type": "short_answer", "question": "...", "answer": "..."}}
]

Context:
{context}"""

# HF Inference Call 
def _call_hf(prompt: str) -> str:
    client = InferenceClient(token=HF_TOKEN)
    response = client.chat_completion(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,  # ← increase this
    )
    return response.choices[0].message.content

# Main Function 
def generate_quiz(notebook_id: str, source_type: str = "all", source_id: str = None) -> dict:
    """
    Full pipeline:
    1. Fetch chunks for notebook (filtered by source type)
    2. Build prompt
    3. Call HF model
    4. Parse quiz JSON
    5. Save to artifacts table
    Returns {"artifact_id": ..., "questions": [...]}
    """
    # 1. Get context
    if source_type == "all":
        chunks = _get_chunks_for_notebook(notebook_id)
    elif source_type == "pdf" and source_id:
        chunks = _get_chunks_by_source_id(notebook_id, source_id)
    elif source_type in ("txt", "url"):
        chunks = _get_chunks_by_type(notebook_id, source_type)
        if not chunks:
            source_label = "text" if source_type == "txt" else "URL"
            raise ValueError(f"No {source_label} sources found. Please add a {source_label} source first.")
    else:
        chunks = _get_chunks_for_notebook(notebook_id)
    
    print(f"Found {len(chunks)} chunks for source_type={source_type}")
    if not chunks:
        raise ValueError("No chunks found for this source. Please add sources first.")

    context = "\n\n".join(chunks[:5])

    # 2. Build prompt
    prompt = _build_prompt(context)

    # 3. Call model
    print(f"Calling HF model for quiz generation...")
    raw_output = _call_hf(prompt)
    print(f"Model response received")

    # 4. Parse
    questions = _parse_quiz(raw_output)
    print(f"Parsed {len(questions)} questions")

    # 5. Save artifact
    artifact = create_artifact(
        notebook_id=notebook_id,
        type="quiz",
        storage_path=json.dumps(questions),
    )

    artifact_id = artifact["id"] if artifact else None
    print(f"Quiz artifact saved: {artifact_id}")

    return {
        "artifact_id": artifact_id,
        "questions": questions,
    }

def _get_chunks_by_source_id(notebook_id: str, source_id: str, limit: int = 10) -> list[str]:
    result = (
        supabase.table("chunks")
        .select("content")
        .eq("notebook_id", notebook_id)
        .eq("source_id", source_id)
        .limit(limit)
        .execute()
    )
    return [row["content"] for row in (result.data or [])]


def _get_chunks_by_type(notebook_id: str, source_type: str, limit: int = 10) -> list[str]:
    result = (
        supabase.table("chunks")
        .select("content, source_id, created_at")
        .eq("notebook_id", notebook_id)
        .order("created_at", desc=True)
        .execute()
    )
    rows = result.data or []
    if source_type == "url":
        filtered = [r for r in rows if r["source_id"].startswith("url_")]
    else:  # txt
        filtered = [r for r in rows if not r["source_id"].startswith("url_") and not r["source_id"].endswith(".pdf")]
    
    if not filtered:
        return []
    
    # Use only the most recent source_id
    latest_source_id = filtered[0]["source_id"]
    return [r["content"] for r in filtered if r["source_id"] == latest_source_id][:limit]

    
def _parse_quiz(raw: str) -> list[dict]:
    print(f"RAW OUTPUT:\n{raw}\n")
    start = raw.find('[')
    if start == -1:
        raise ValueError("No JSON array found in model output.")
    
    json_str = raw[start:].strip()
    
    # Try parsing as-is first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Try fixing truncated JSON by extracting complete objects only
    objects = re.findall(r'\{[^{}]+\}', json_str, re.DOTALL)
    if objects:
        try:
            return json.loads('[' + ','.join(objects) + ']')
        except json.JSONDecodeError:
            pass
    
    raise ValueError("Could not parse quiz JSON from model output.")