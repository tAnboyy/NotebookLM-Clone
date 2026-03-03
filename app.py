from pathlib import Path
import shutil
import sys
import warnings

# Flush print immediately
def _log(msg):
    print(msg, flush=True)

_log("1. Loading env...")
# Suppress noisy dependency warnings
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*chardet.*")

from dotenv import load_dotenv

# Load .env from project root (parent of NotebookLM-Clone) so HF_TOKEN etc. are available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

_log("2. Loading Gradio...")
from datetime import datetime
import gradio as gr
_log("2a. Loading gradio_client...")
import gradio_client.utils as gradio_client_utils

_log("3. Loading backend...")
from backend.ingestion_service import ingest_pdf_chunks, ingest_url_chunks, remove_chunks_for_source
from backend.notebook_service import create_notebook, list_notebooks, rename_notebook, delete_notebook
from backend.podcast_service import generate_podcast, generate_podcast_audio
from backend.chat_service import load_chat
from backend.rag_service import rag_chat
from backend.report_service import generate_report

import hashlib
_log("4. Imports done.")

_original_gradio_get_type = gradio_client_utils.get_type
_original_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type


def _patched_gradio_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _original_gradio_get_type(schema)


def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)


gradio_client_utils.get_type = _patched_gradio_get_type
gradio_client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

# Theme: adapts to light/dark mode (use default font to avoid network fetch on startup)
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

CUSTOM_CSS = """
.gradio-container { max-width: 1000px !important; margin: 0 auto !important; }
.container { max-width: 1000px; margin: 0 auto; padding: 0 24px; }

.header-bar { padding: 12px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 24px; display: flex !important; justify-content: space-between !important; align-items: center !important; white-space: nowrap; }
.login-center { display: flex; justify-content: center; width: 100%; }
#auth-text { white-space: nowrap; margin: 8px 0 16px 0; font-size: 0.95rem; opacity: 0.9; }
.gr-button { padding: 14px 28px !important; font-size: 0.9rem !important; border-radius: 12px !important; white-space: nowrap !important; width: auto !important; }
.gr-button[aria-label*="Logout"] { min-width: auto !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; }
.header-bar .gr-button { padding-left: 40px !important; padding-right: 40px !important; min-width: 220px !important; font-size: 0.8rem !important; }
.dark .header-bar { border-bottom: 1px solid #334155; }

.hero-section { margin-bottom: 16px; }
.login-container { padding: 12px 0; }
.create-strip { padding: 18px; border-radius: 16px; }
.create-row { display: flex !important; align-items: center !important; gap: 16px !important; }
.create-label { white-space: nowrap; font-size: 0.95rem; margin: 0; min-width: 180px; }
.create-row .gr-textbox { flex: 1 !important; }
.create-row .gr-textbox textarea,
.create-row .gr-textbox input { border-radius: 10px !important; }
.create-row .gr-button { border-radius: 10px !important; padding: 10px 20px !important; }
.hero-title { font-size: 2rem; font-weight: 700; color: #1e293b; margin: 0 0 8px 0; }
.hero-sub { font-size: 1rem; color: #64748b; margin: 0; line-height: 1.5; }

.section-card { padding: 24px; border-radius: 16px; background: #f8fafc; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.notebook-card { padding: 14px 20px; border-radius: 12px; background: #fff; margin-bottom: 8px; border: 1px solid #e2e8f0; display: flex; align-items: center; gap: 12px; transition: background 0.15s ease; }
.notebook-card:hover { background: #f8fafc; }

.section-title { font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0 0 16px 0; }
.section-row { display: flex !important; align-items: center !important; gap: 16px !important; margin-bottom: 12px; }
.section-row .gr-textbox { flex: 1 !important; }
.section-row .gr-button { border-radius: 10px !important; padding: 10px 20px !important; }

.status { font-size: 0.875rem; color: #64748b; margin-top: 16px; padding: 12px 16px; background: #f1f5f9; border-radius: 12px; }

@media (prefers-color-scheme: dark) {
  .hero-title { color: #f1f5f9 !important; }
  .hero-sub { color: #94a3b8 !important; }
  .section-card { background: #1e293b !important; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
  .section-title { color: #f1f5f9 !important; }
  .notebook-card { background: #334155 !important; border-color: #475569; }
  .notebook-card:hover { background: #475569 !important; }
  .status { color: #94a3b8 !important; background: #334155 !important; }
}
.dark .hero-title { color: #f1f5f9 !important; }
.dark .hero-sub { color: #94a3b8 !important; }
.dark .section-card { background: #1e293b !important; }
.dark .section-title { color: #f1f5f9 !important; }
.dark .notebook-card { background: #334155 !important; border-color: #475569; }
.dark .notebook-card:hover { background: #475569 !important; }
.dark .status { color: #94a3b8 !important; background: #334155 !important; }
"""

def _user_id(profile: gr.OAuthProfile | None) -> str | None:
    """Extract user_id from HF OAuth profile. None if not logged in."""
    if not profile:
        return None
    return (
        getattr(profile, "id", None)
        or getattr(profile, "sub", None)
        or getattr(profile, "preferred_username", None)
        or getattr(profile, "username", None)
        or getattr(profile, "name", None)
    )


def _get_notebooks(user_id: str | None):
    if not user_id:
        return []
    return list_notebooks(user_id)


def _safe_create(new_name, state, selected_id, profile: gr.OAuthProfile | None = None):
    """Create notebook with name from text box."""
    try:
        user_id = _user_id(profile)
        if not user_id:
            return gr.skip(), gr.skip(), gr.skip(), "Please sign in with Hugging Face"
        name = (new_name or "").strip() or "Untitled Notebook"
        nb = create_notebook(user_id, name)
        if nb:
            notebooks = _get_notebooks(user_id)
            new_state = [(n["notebook_id"], n["name"]) for n in notebooks]
            status = f"Created: {nb['name']}"
            return "", new_state, nb["notebook_id"], status
        return gr.skip(), gr.skip(), gr.skip(), "Failed to create"
    except Exception as e:
        return gr.skip(), gr.skip(), gr.skip(), f"Error: {e}"


def _safe_rename(idx, new_name, state, selected_id, profile: gr.OAuthProfile | None = None):
    """Rename notebook at index."""
    try:
        if idx is None or idx < 0 or idx >= len(state):
            return gr.skip(), gr.skip(), "Invalid selection"
        nb_id, _ = state[idx]
        name = (new_name or "").strip()
        if not name:
            return gr.skip(), gr.skip(), "Enter a name."
        user_id = _user_id(profile)
        if not user_id:
            return gr.skip(), gr.skip(), "Please sign in"
        ok = rename_notebook(user_id, nb_id, name)
        if ok:
            notebooks = _get_notebooks(user_id)
            new_state = [(n["notebook_id"], n["name"]) for n in notebooks]
            return new_state, selected_id, f"Renamed to: {name}"
        return gr.skip(), gr.skip(), "Failed to rename"
    except Exception as e:
        return gr.skip(), gr.skip(), f"Error: {e}"


def _safe_delete(idx, state, selected_id, profile: gr.OAuthProfile | None = None):
    """Delete notebook at index."""
    try:
        if idx is None or idx < 0 or idx >= len(state):
            return gr.skip(), gr.skip(), "Invalid selection"
        nb_id, _ = state[idx]
        user_id = _user_id(profile)
        if not user_id:
            return gr.skip(), gr.skip(), "Please sign in"
        ok = delete_notebook(user_id, nb_id)
        if ok:
            notebooks = _get_notebooks(user_id)
            new_state = [(n["notebook_id"], n["name"]) for n in notebooks]
            new_selected = notebooks[0]["notebook_id"] if notebooks else None
            return new_state, new_selected, "Notebook deleted"
        return gr.skip(), gr.skip(), "Failed to delete"
    except Exception as e:
        return gr.skip(), gr.skip(), f"Error: {e}"


def _initial_load(profile: gr.OAuthProfile | None = None):
    """Load notebooks on app load. Uses HF OAuth profile for user_id."""
    user_id = _user_id(profile)
    notebooks = _get_notebooks(user_id)
    state = [(n["notebook_id"], n["name"]) for n in notebooks]
    selected = notebooks[0]["notebook_id"] if notebooks else None
    status = f"Signed in as {user_id}" if user_id else "Sign in with Hugging Face to manage notebooks."
    auth_update = f"You are logged in as {getattr(profile, 'name', None) or user_id} ({_user_id(profile)})" if user_id else ""
    auth_row_visible = bool(user_id)
    return state, selected, status, auth_update, gr.update(visible=auth_row_visible), gr.update(visible=bool(user_id)), gr.update(visible=not bool(user_id))


REPORT_SCOPE_LABELS = {
    "All sources (PDFs, URLs, text)": "all",
    "PDF uploads only": "pdf",
    "Web URLs only": "url",
    "Uploaded text only": "text",
}

REPORT_SCOPE_DESCRIPTIONS = {
    "all": "PDFs, URLs, and uploaded text",
    "pdf": "uploaded PDFs",
    "url": "ingested web URLs",
    "text": "uploaded text files",
}

DEFAULT_REPORT_SCOPE_LABEL = "All sources (PDFs, URLs, text)"


def _resolve_report_scope(label: str) -> tuple[str, str]:
    value = REPORT_SCOPE_LABELS.get(label, "all")
    desc = REPORT_SCOPE_DESCRIPTIONS.get(value, "selected sources")
    return value, desc


def _generate_report(scope_label, notebook_id, profile: gr.OAuthProfile | None):
    scope_value, scope_desc = _resolve_report_scope(scope_label)
    user_id = _user_id(profile)
    if not user_id:
        return "Please sign in with Hugging Face before generating a report.", ""
    if not notebook_id:
        return "Select a notebook first to generate a report.", ""
    try:
        report_text = generate_report(notebook_id, scope_value)
        status = f"Report ready for {scope_desc}."
        return status, report_text
    except ValueError as error:
        return f"⚠️ {error}", ""
    except Exception as error:
        return f"Error generating report: {error}", ""


def _safe_upload_pdfs(files, selected_id, profile: gr.OAuthProfile | None):
    """Upload PDF files for the selected notebook."""
    try:
        user_id = _user_id(profile)
        if not user_id:
            return "Please sign in with Hugging Face before uploading PDFs."
        if not selected_id:
            return "Select a notebook first, then upload PDFs."
        if not files:
            return "Choose at least one PDF to upload."

        if isinstance(files, str):
            file_paths = [files]
        else:
            file_paths = []
            for file_item in files:
                file_path = getattr(file_item, "name", file_item)
                if file_path:
                    file_paths.append(file_path)

        if not file_paths:
            return "No files were received. Try uploading again."

        target_dir = Path("data") / "uploads" / user_id / str(selected_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        uploaded = []
        total_chunks = 0
        for file_path in file_paths:
            source_path = Path(file_path)
            if source_path.suffix.lower() != ".pdf":
                continue

            destination = target_dir / source_path.name
            if destination.exists():
                index = 1
                while True:
                    candidate = target_dir / f"{source_path.stem}_{index}{source_path.suffix}"
                    if not candidate.exists():
                        destination = candidate
                        break
                    index += 1

            shutil.copy2(source_path, destination)
            uploaded.append(destination.name)
            total_chunks += ingest_pdf_chunks(str(selected_id), destination.name, destination)

        if not uploaded:
            return "Only .pdf files are allowed."

        return f"Uploaded {len(uploaded)} PDF(s): {', '.join(uploaded)}. Indexed {total_chunks} chunk(s) for RAG."
    except Exception as error:
        return f"Error uploading PDFs: {error}"


def _list_uploaded_pdfs(selected_id, profile: gr.OAuthProfile | None = None):
    """List uploaded PDFs for the selected notebook."""
    user_id = _user_id(profile)
    if not user_id or not selected_id:
        return gr.update(choices=[], value=None)

    target_dir = Path("data") / "uploads" / user_id / str(selected_id)
    if not target_dir.exists():
        return gr.update(choices=[], value=None)

    pdf_names = sorted([path.name for path in target_dir.glob("*.pdf")])
    selected_name = pdf_names[0] if pdf_names else None
    return gr.update(choices=pdf_names, value=selected_name)


def _safe_remove_pdf(file_name, selected_id, profile: gr.OAuthProfile | None = None):
    """Remove one uploaded PDF from the selected notebook."""
    try:
        user_id = _user_id(profile)
        if not user_id:
            return "Please sign in with Hugging Face before removing PDFs."
        if not selected_id:
            return "Select a notebook first."
        if not file_name:
            return "Select a PDF to remove."

        safe_name = Path(file_name).name
        target_file = Path("data") / "uploads" / user_id / str(selected_id) / safe_name
        if not target_file.exists() or target_file.suffix.lower() != ".pdf":
            return "Selected PDF was not found."

        target_file.unlink()
        remove_chunks_for_source(str(selected_id), safe_name)
        return f"Removed PDF: {safe_name}"
    except Exception as error:
        return f"Error removing PDF: {error}"
    
def _url_source_id(url: str) -> str:
    """Stable source_id so re-ingesting the same URL overwrites old chunks."""
    h = hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"url_{h}"


def _safe_ingest_url(url, selected_id, profile: gr.OAuthProfile | None = None):
    """Ingest one URL into chunks table for the selected notebook."""
    try:
        user_id = _user_id(profile)
        if not user_id:
            return "", "Please sign in with Hugging Face before ingesting a URL."
        if not selected_id:
            return "", "Select a notebook first, then ingest a URL."

        cleaned = (url or "").strip()
        if not cleaned:
            return "", "Enter a URL."
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            return "", "URL must start with http:// or https://"

        source_id = _url_source_id(cleaned)
        chunk_count = ingest_url_chunks(str(selected_id), source_id, cleaned)

        if chunk_count == 0:
            return "", (
                "Ingested URL but extracted 0 chunks. Page may be JS-rendered/blocked/non-text. "
                "Try a simpler static page (example.com / Wikipedia)."
            )

        return "", f"Ingested URL. Indexed {chunk_count} chunk(s). Source: {cleaned}"
    except Exception as error:
        return "", f"Error ingesting URL: {error}"
    
def _safe_remove_url(url, selected_id, profile: gr.OAuthProfile | None = None):
    try:
        user_id = _user_id(profile)
        if not user_id:
            return "", "Please sign in with Hugging Face before ingesting a URL."
        if not selected_id:
            return "", "Select a notebook first, then remove a URL."
        
        cleaned = (url or "").strip()
        if not cleaned:
            return "", "Enter a URL."
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            return "", "URL must start with http:// or https://"

        source_id = _url_source_id(cleaned)
        remove_chunks_for_source(str(selected_id), source_id)
        return "", f"Removed URL: {cleaned}"
    except Exception as error:
        return "", f"Error removing URL: {error}"



# ── Upload Handler Functions ──────────────────────────────────
def _do_upload(text_content, title, notebook_id, profile: gr.OAuthProfile | None):
    """Handle direct text input and ingestion."""
    from backend.ingestion_txt import ingest_txt

    user_id = _user_id(profile)

    if not user_id:
        return "Please sign in first."
    if not notebook_id:
        return "Please select a notebook first."
    if not text_content or not text_content.strip():
        return "No text entered."

    try:
        filename = (title or "").strip()
        if not filename:
            filename = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not filename.endswith(".txt"):
            filename = filename + ".txt"

        file_bytes = text_content.encode("utf-8")

        result = ingest_txt(
            file_bytes=file_bytes,
            filename=filename,
            notebook_id=notebook_id,
            user_id=user_id
        )

        meta = result["metadata"]
        return (
            f" **{result['filename']}** saved successfully!\n\n"
            f"- Size: {meta['size_bytes'] / 1024:.1f} KB"
        )

    except ValueError as e:
        return f" {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "No sources yet."
    lines = ["| Filename | Type | Status | Words |",
             "|----------|------|--------|-------|"]
    for s in sources:
        meta = s.get("metadata") or {}
        words = meta.get("word_count", "—")
        lines.append(f"| {s['filename']} | {s['file_type']} | {s['status']} | {words} |")
    return "\n".join(lines)


def _load_sources(notebook_id, profile: gr.OAuthProfile | None):
    from backend.ingestion_txt import list_sources
    if not notebook_id:
        return ""
    sources = list_sources(notebook_id)
    return _format_sources(sources)


def _safe_generate_podcast(notebook_id, profile: gr.OAuthProfile | None = None):
    user_id = _user_id(profile)
    if not user_id:
        return "Please sign in first.", ""
    if not notebook_id:
        return "Please select a notebook first.", ""

    try:
        result = generate_podcast(notebook_id=str(notebook_id), user_id=user_id)
        status = (
            f"Podcast generated. Artifact: {result['artifact_id'] or 'saved'} | "
            f"Sources: {result['sources_count']} | Chunks: {result['chunks_used']}"
        )
        return status, result["script"]
    except Exception as error:
        return f"Error generating podcast: {error}", ""


def _safe_generate_podcast_audio(notebook_id, script, profile: gr.OAuthProfile | None = None):
    user_id = _user_id(profile)
    if not user_id:
        return "Please sign in first.", None
    if not notebook_id:
        return "Please select a notebook first.", None
    if not script or not script.strip():
        return "Generate a podcast script first.", None

    try:
        result = generate_podcast_audio(notebook_id=str(notebook_id), user_id=user_id, script=script)
        status = f"Podcast audio generated. Artifact: {result['artifact_id'] or 'saved'}"
        return status, result["audio_path"]
    except Exception as error:
        return f"Error generating podcast audio: {error}", None

# Quiz Handlers 
def _get_notebook_pdfs(notebook_id):
    if not notebook_id:
        return gr.update(choices=[], value=None, visible=False)
    from backend.db import supabase
    result = (
        supabase.table("chunks")
        .select("source_id")
        .eq("notebook_id", notebook_id)
        .execute()
    )
    pdfs = list({r["source_id"] for r in (result.data or []) if r["source_id"].endswith(".pdf")})
    return gr.update(choices=pdfs, value=pdfs[0] if pdfs else None, visible=True)

def _generate_quiz(notebook_id, source_type, pdf_source_id, profile: gr.OAuthProfile | None):
    from backend.quiz_service import generate_quiz

    user_id = _user_id(profile)
    if not user_id:
        return "Please sign in first.", [], *([gr.update(visible=False)] * 5 * 4), gr.update(visible=False), ""
    if not notebook_id:
        return "Please select a notebook first.", [], *([gr.update(visible=False)] * 5 * 4), gr.update(visible=False), ""

    type_map = {"Text": "txt", "PDF": "pdf", "URL": "url", "All": "all"}
    source_type_key = type_map.get(source_type, "all")

    try:
        result = generate_quiz(notebook_id, source_type=source_type_key, source_id=pdf_source_id)
        questions = result["questions"]
        updates = []
        for i in range(5):
            if i < len(questions):
                q = questions[i]
                q_label = f"**Q{i+1}. {q['question']}**"
                if q["type"] == "multiple_choice":
                    updates += [gr.update(visible=True), gr.update(value=q_label), gr.update(choices=q["options"], value=None, visible=True), gr.update(value="", visible=False)]
                elif q["type"] == "true_false":
                    updates += [gr.update(visible=True), gr.update(value=q_label), gr.update(choices=["True", "False"], value=None, visible=True), gr.update(value="", visible=False)]
                else:
                    updates += [gr.update(visible=True), gr.update(value=q_label), gr.update(choices=[], value=None, visible=False), gr.update(value="", visible=True)]
            else:
                updates += [gr.update(visible=False), gr.update(value=""), gr.update(choices=[], value=None, visible=False), gr.update(value="", visible=False)]
        return "Quiz generated!", questions, *updates, gr.update(visible=True), ""
    except Exception as e:
        return f" {e}", [], *([gr.update(visible=False)] * 5 * 4), gr.update(visible=False), ""


def _submit_quiz(questions, *answers):
    if not questions:
        return " No quiz loaded."
    score = 0
    lines = []
    for i, q in enumerate(questions):
        radio_ans = answers[i] or ""
        text_ans = answers[i + 5] or ""
        user_ans = text_ans.strip() if q["type"] == "short_answer" else radio_ans.strip()
        correct = q["answer"].strip()

        if not user_ans:
            is_correct = False
        elif q["type"] == "multiple_choice":
            user_letter = user_ans.split(".")[0].strip().upper()
            correct_letter = correct[0].upper()
            is_correct = user_letter == correct_letter
        elif q["type"] == "true_false":
            is_correct = user_ans.lower() == correct.lower()
        else:
            is_correct = user_ans.lower() in correct.lower() or correct.lower() in user_ans.lower()

        if is_correct:
            score += 1
            lines.append(f"✅ **Q{i+1}**: Correct! *(Answer: {correct})*")
        else:
            lines.append(f"❌ **Q{i+1}**: Incorrect. *(Your answer: {user_ans or 'blank'} | Correct: {correct})*")

    lines.append(f"\n**Score: {score}/{len(questions)}**")
    return "\n\n".join(lines)
def _chat_history_to_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    """Convert load_chat output to Gradio Chatbot format [(user, assistant), ...]."""
    pairs = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "user":
            user_content = m["content"] or ""
            asst_content = ""
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                asst_content = messages[i + 1]["content"] or ""
                i += 1
            pairs.append((user_content, asst_content))
        i += 1
    return pairs


def _load_chat_history(notebook_id) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Load chat for notebook. Returns (history_pairs, history_pairs) for State and Chatbot."""
    if not notebook_id:
        return [], []
    messages = load_chat(notebook_id)
    pairs = _chat_history_to_pairs(messages)
    return pairs, pairs


def _on_chat_submit(query, notebook_id, chat_history, profile: gr.OAuthProfile | None):
    """Handle chat submit: call RAG, return updated history."""
    if not notebook_id:
        return "", chat_history, "Select a notebook first."
    if not query or not query.strip():
        return "", chat_history, "Enter a message."
    user_id = _user_id(profile)
    if not user_id:
        return "", chat_history, "Please sign in first."
    try:
        answer, updated = rag_chat(notebook_id, query.strip(), chat_history)
        return "", updated, ""
    except Exception as e:
        return "", chat_history, f"Error: {e}"

with gr.Blocks(
    title="NotebookLM Clone - Notebooks",
    theme=theme,
    css=CUSTOM_CSS,
) as demo:
    with gr.Row(elem_classes=["header-bar"]):
        gr.Markdown("### 📓 NotebookLM Clone")
        login_btn = gr.LoginButton(value="🤗 Login with Hugging Face", size="lg")

    with gr.Row(visible=False) as auth_info_row:
        auth_text = gr.Markdown("", elem_id="auth-text")

    gr.HTML("""
    <div class="container hero-section">
        <h1 class="hero-title">📓 NotebookLM Clone</h1>
        <p class="hero-sub">Chat with your documents. Generate reports, quizzes, and podcasts with citations.</p>
    </div>
    """)

    with gr.Column(visible=False, elem_classes=["login-container"]) as login_container:
        gr.Markdown("**Sign in with Hugging Face to access your notebooks.**", elem_classes=["login-center"])

    with gr.Column(visible=False) as app_content:
        nb_state = gr.State([])
        selected_notebook_id = gr.State(None)

        with gr.Group(elem_classes=["create-strip"]):
            with gr.Row(elem_classes=["create-row"]):
                gr.Markdown("Create new notebook", elem_classes=["create-label"])
                create_txt = gr.Textbox(
                    placeholder="Enter new notebook name",
                    show_label=False,
                    container=False,
                    value="",
                )
                create_btn = gr.Button("Create", variant="primary", size="sm")

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("**Sources**", elem_classes=["section-title"])
            gr.Markdown("*Upload PDFs, ingest URLs, or add text to your selected notebook*")
            with gr.Row(elem_classes=["section-row"]):
                pdf_upload_btn = gr.UploadButton(
                    "Upload PDFs",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                    variant="secondary",
                )
            with gr.Row(elem_classes=["section-row"]):
                uploaded_pdf_dd = gr.Dropdown(
                    label="Uploaded PDFs",
                    choices=[],
                    value=None,
                    scale=3,
                    allow_custom_value=False,
                )
                remove_pdf_btn = gr.Button("Remove selected PDF", variant="stop", scale=1)
            with gr.Row(elem_classes=["section-row"]):
                url_txt = gr.Textbox(
                    label="Ingest web URL",
                    placeholder="https://example.com",
                    value="",
                    scale=3,
                )
                ingest_url_btn = gr.Button("Ingest URL", variant="primary", scale=1)
                remove_url_btn = gr.Button("Delete URL", variant="stop", scale=1)

        gr.HTML("<br>")
        gr.Markdown("**Your Notebooks**", elem_classes=["section-title"])
        gr.Markdown("*Selected notebook is used for chat and ingestion*", elem_id="sub-hint")
        gr.HTML("<br>")

        status = gr.Markdown("Sign in with Hugging Face to manage notebooks.", elem_classes=["status"])

        @gr.render(inputs=[nb_state])
        def render_notebooks(state):
            if not state:
                gr.Markdown("No notebooks yet. Create one to get started.")
            else:
                for i, (nb_id, name) in enumerate(state):
                    idx = i
                    with gr.Row(elem_classes=["notebook-card"]):
                        name_txt = gr.Textbox(value=name, show_label=False, scale=4, min_width=240, key=f"nb-name-{nb_id}")
                        select_btn = gr.Button("Select", variant="primary", scale=1, min_width=80, size="sm")
                        rename_btn = gr.Button("Rename", variant="secondary", scale=1, min_width=80, size="sm")
                        delete_btn = gr.Button("Delete", variant="secondary", scale=1, min_width=80, size="sm")

                        def on_select(nb_id=nb_id):
                            return nb_id

                        def on_select_status():
                            return "Selected notebook updated. Use this for chat/ingestion."

                        select_btn.click(
                            on_select,
                            inputs=None,
                            outputs=[selected_notebook_id],
                        ).then(on_select_status, None, [status]).then(_list_uploaded_pdfs, inputs=[selected_notebook_id], outputs=[uploaded_pdf_dd])

                        rename_btn.click(
                            _safe_rename,
                            inputs=[gr.State(idx), name_txt, nb_state, selected_notebook_id],
                            outputs=[nb_state, selected_notebook_id, status],
                            api_name=False,
                        )

                        delete_btn.click(
                            _safe_delete,
                            inputs=[gr.State(idx), nb_state, selected_notebook_id],
                            outputs=[nb_state, selected_notebook_id, status],
                            api_name=False,
                        ).then(_list_uploaded_pdfs, inputs=[selected_notebook_id], outputs=[uploaded_pdf_dd])

        gr.HTML("<br>")

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("**Add Text**", elem_classes=["section-title"])
            gr.Markdown("*Select a notebook above, then paste or type your text*")
            with gr.Row():
                txt_title = gr.Textbox(
                    label="Title",
                    placeholder="Give this text a name (e.g. 'Lecture Notes Week 1')",
                    scale=1,
                )
            txt_input = gr.Textbox(
                label="Text Content",
                placeholder="Paste or type your text here...",
                lines=10,
            )
            submit_btn = gr.Button("Save & Process", variant="primary")
            upload_status = gr.Markdown("", elem_classes=["status"])
            sources_display = gr.Markdown("")

        gr.HTML("<br>")
        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("**Reports**", elem_classes=["section-title"])
            gr.Markdown("*Generate a concise report based on your uploaded PDFs, ingested URLs, or added text.*")
            with gr.Row(elem_classes=["section-row"]):
                report_scope_dd = gr.Dropdown(
                    label="Report scope",
                    choices=list(REPORT_SCOPE_LABELS.keys()),
                    value=DEFAULT_REPORT_SCOPE_LABEL,
                    scale=3,
                )
                report_btn = gr.Button("Generate report", variant="primary", scale=1)
            report_status = gr.Markdown("Select a scope and click generate.", elem_classes=["status"])
            report_output = gr.Markdown("", elem_id="report-output")

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("**Chat**", elem_classes=["section-title"])
            gr.Markdown("*Ask questions about your notebook sources. Answers are grounded in retrieved chunks with citations.*")
            chat_history_state = gr.State([])
            chatbot = gr.Chatbot(label="Chat history", height=400)
            chat_input = gr.Textbox(
                label="Message",
                placeholder="Ask a question about your sources...",
                show_label=False,
                lines=2,
            )
            chat_submit_btn = gr.Button("Send", variant="primary")
            chat_status = gr.Markdown("", elem_classes=["status"])

    demo.load(
        _initial_load,
        inputs=None,
        outputs=[nb_state, selected_notebook_id, status, auth_text, auth_info_row, app_content, login_container],
        api_name=False,
    )
    demo.load(_list_uploaded_pdfs, inputs=[selected_notebook_id], outputs=[uploaded_pdf_dd], api_name=False)

    def _on_notebook_select_for_chat(notebook_id):
        hist, _ = _load_chat_history(notebook_id)
        return hist, hist

    selected_notebook_id.change(
        _on_notebook_select_for_chat,
        inputs=[selected_notebook_id],
        outputs=[chat_history_state, chatbot],
        api_name=False,
    )

    create_btn.click(
        _safe_create,
        inputs=[create_txt, nb_state, selected_notebook_id],
        outputs=[create_txt, nb_state, selected_notebook_id, status],
        api_name=False,
    ).then(_list_uploaded_pdfs, inputs=[selected_notebook_id], outputs=[uploaded_pdf_dd])

    pdf_upload_btn.upload(
        _safe_upload_pdfs,
        inputs=[pdf_upload_btn, selected_notebook_id],
        outputs=[status],
        api_name=False,
    ).then(_list_uploaded_pdfs, inputs=[selected_notebook_id], outputs=[uploaded_pdf_dd])

    ingest_url_btn.click(
        _safe_ingest_url,
        inputs=[url_txt, selected_notebook_id],
        outputs=[url_txt, status],
        api_name=False,
    )

    remove_url_btn.click(
        _safe_remove_url,
        inputs=[url_txt, selected_notebook_id],
        outputs=[url_txt, status],
        api_name=False
    )

    remove_pdf_btn.click(
        _safe_remove_pdf,
        inputs=[uploaded_pdf_dd, selected_notebook_id],
        outputs=[status],
        api_name=False,
    ).then(_list_uploaded_pdfs, inputs=[selected_notebook_id], outputs=[uploaded_pdf_dd])

    # Per-row: Rename, Delete, Select (profile injected by Gradio for OAuth)
    for i in range(MAX_NOTEBOOKS):
        rename_btn = row_components[i]["rename"]
        delete_btn = row_components[i]["delete"]
        select_btn = row_components[i]["select"]
        name_txt = row_components[i]["name"]

        rename_btn.click(
            _safe_rename,
            inputs=[gr.State(i), name_txt, nb_state, selected_notebook_id],
            outputs=[nb_state, selected_notebook_id, status] + row_outputs,
            api_name=False,
        )
        delete_btn.click(
            _safe_delete,
            inputs=[gr.State(i), nb_state, selected_notebook_id],
            outputs=[nb_state, selected_notebook_id, status] + row_outputs,
            api_name=False,
        ).then(_list_uploaded_pdfs, inputs=[selected_notebook_id], outputs=[uploaded_pdf_dd])
        def _on_select():
            return "Selected notebook updated. Use this for chat/ingestion."
        select_btn.click(
            _select_notebook,
            inputs=[gr.State(i), nb_state],
            outputs=[selected_notebook_id],
            api_name=False,
        ).then(_on_select, None, [status]).then(_list_uploaded_pdfs, inputs=[selected_notebook_id], outputs=[uploaded_pdf_dd])

    
    # Text Input Section 
    gr.Markdown("---")
    gr.Markdown("## Add Text")
    gr.Markdown("Select a notebook above, then paste or type your text.")

    with gr.Row():
        txt_title = gr.Textbox(
            label="Title",
            placeholder="Give this text a name (e.g. 'Lecture Notes Week 1')",
            scale=1,
        )

    txt_input = gr.Textbox(
        label="Text Content",
        placeholder="Paste or type your text here...",
        lines=10,
    )

    submit_btn = gr.Button("Save & Process", variant="primary")

    upload_status = gr.Markdown("", elem_classes=["status"])

    # Podcast Section
    gr.Markdown("---")
    gr.Markdown("## Podcast")
    gr.Markdown("Generate a podcast script for the selected notebook using all ingested content.")
    with gr.Row():
        podcast_btn = gr.Button("Generate Podcast", variant="primary")
        podcast_audio_btn = gr.Button("Generate Podcast Audio", variant="secondary")
    podcast_status = gr.Markdown("", elem_classes=["status"])
    podcast_script = gr.Markdown("")
    podcast_audio = gr.Audio(label="Podcast Audio", type="filepath")

   # Quiz Section 
    gr.Markdown("---")
    gr.Markdown("## Generate Quiz")
    gr.Markdown("Select a source type then generate a quiz.")

    quiz_source_type = gr.Radio(
        choices=["Text", "PDF", "URL", "All"],
        value="All",
        label="Source type",
    )
    quiz_pdf_dd = gr.Dropdown(
        label="Select PDF",
        choices=[],
        value=None,
        visible=False,
    )
    generate_quiz_btn = gr.Button("Generate Quiz", variant="primary")
    quiz_status = gr.Markdown("")
    quiz_state = gr.State([])

    quiz_components = []
    for i in range(5):
        with gr.Group(visible=False) as q_group:
            q_text = gr.Markdown("")
            q_radio = gr.Radio(choices=[], label="Your answer", visible=False)
            q_textbox = gr.Textbox(label="Your answer", visible=False)
        quiz_components.append({"group": q_group, "text": q_text, "radio": q_radio, "textbox": q_textbox})

    submit_quiz_btn = gr.Button("Submit Answers", variant="secondary", visible=False)
    quiz_results = gr.Markdown("")

    submit_btn.click(
        _do_upload,
        inputs=[txt_input, txt_title, selected_notebook_id],
        outputs=[upload_status],
    )

    report_btn.click(
        _generate_report,
        inputs=[report_scope_dd, selected_notebook_id],
        outputs=[report_status, report_output],
        api_name=False,
    )

    selected_notebook_id.change(
        _load_sources,
        inputs=[selected_notebook_id],
        outputs=[podcast_status, podcast_script],
        api_name=False,
    )

    podcast_btn.click(
        _safe_generate_podcast,
        inputs=[selected_notebook_id],
        outputs=[podcast_status, podcast_script],
        api_name=False,
    )

    podcast_audio_btn.click(
        _safe_generate_podcast_audio,
        inputs=[selected_notebook_id, podcast_script],
        outputs=[podcast_status, podcast_audio],
        api_name=False,
    )

    quiz_source_type.change(
        lambda t, nb: _get_notebook_pdfs(nb) if t == "PDF" else gr.update(visible=False, choices=[], value=None),
        inputs=[quiz_source_type, selected_notebook_id],
        outputs=[quiz_pdf_dd],
    )

    quiz_all_outputs = [quiz_status, quiz_state]
    for c in quiz_components:
        quiz_all_outputs += [c["group"], c["text"], c["radio"], c["textbox"]]
    quiz_all_outputs += [submit_quiz_btn, quiz_results]

    generate_quiz_btn.click(
        _generate_quiz,
        inputs=[selected_notebook_id, quiz_source_type, quiz_pdf_dd],
        outputs=quiz_all_outputs,
        api_name=False,
    )

    submit_quiz_btn.click(
        _submit_quiz,
        inputs=[quiz_state] + [c["radio"] for c in quiz_components] + [c["textbox"] for c in quiz_components],
        outputs=[quiz_results],
        api_name=False,
    )

    chat_submit_btn.click(
        _on_chat_submit,
        inputs=[chat_input, selected_notebook_id, chat_history_state],
        outputs=[chat_input, chat_history_state, chat_status],
        api_name=False,
    ).then(
        lambda h: (h, h),
        inputs=[chat_history_state],
        outputs=[chat_history_state, chatbot],
    )

if __name__ == "__main__":
    _log("5. Launching Gradio...")
    demo.launch()
