from pathlib import Path
import shutil

from dotenv import load_dotenv

# Load .env from project root (parent of NotebookLM-Clone) so HF_TOKEN etc. are available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

from datetime import datetime
import gradio as gr
import gradio_client.utils as gradio_client_utils

from backend.ingestion_service import ingest_pdf_chunks, ingest_url_chunks, remove_chunks_for_source
from backend.notebook_service import create_notebook, list_notebooks, rename_notebook, delete_notebook

import hashlib

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

# Theme: adapts to light/dark mode
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

CUSTOM_CSS = """
.container { max-width: 720px; margin: 0 auto; padding: 0 24px; }
.login-center { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; padding: 24px 0; }
.login-center .login-btn-wrap { display: flex; justify-content: center; width: 100%; }
.login-center .login-btn-wrap button { display: inline-flex; align-items: center; gap: 8px; }
.hero { font-size: 1.5rem; font-weight: 600; color: #1e293b; margin-bottom: 8px; }
.sub { font-size: 0.875rem; color: #64748b; margin-bottom: 24px; }
.nb-row { display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid #e2e8f0; }
.nb-row:last-child { border-bottom: none; }
.gr-button { min-height: 36px !important; padding: 0 16px !important; font-weight: 500 !important; border-radius: 8px !important; }
.gr-input { min-height: 40px !important; border-radius: 8px !important; }
.status { font-size: 0.875rem; color: #64748b; margin-top: 16px; padding: 12px 16px; background: #f8fafc; border-radius: 8px; }
@media (prefers-color-scheme: dark) {
  .hero { color: #f1f5f9 !important; }
  .sub { color: #94a3b8 !important; }
  .nb-row { border-color: #334155 !important; }
  .status { color: #94a3b8 !important; background: #1e293b !important; }
}
.dark .hero { color: #f1f5f9 !important; }
.dark .sub { color: #94a3b8 !important; }
.dark .nb-row { border-color: #334155 !important; }
.dark .status { color: #94a3b8 !important; background: #1e293b !important; }
"""

MAX_NOTEBOOKS = 20


def _user_id(profile: gr.OAuthProfile | None) -> str | None:
    """Extract user_id from HF OAuth profile. None if not logged in."""
    return profile.name if profile else None


def _get_notebooks(user_id: str | None):
    if not user_id:
        return []
    return list_notebooks(user_id)


def _safe_create(new_name, state, selected_id, profile: gr.OAuthProfile | None):
    """Create notebook with name from text box."""
    try:
        user_id = _user_id(profile)
        if not user_id:
            return gr.skip(), gr.skip(), gr.skip(), "Please sign in with Hugging Face", *([gr.skip()] * (MAX_NOTEBOOKS * 2))
        name = (new_name or "").strip() or "Untitled Notebook"
        nb = create_notebook(user_id, name)
        if nb:
            notebooks = _get_notebooks(user_id)
            state = [(n["notebook_id"], n["name"]) for n in notebooks]
            updates = _build_row_updates(notebooks)
            new_selected = nb["notebook_id"]
            status = f"Created: {nb['name']}"
            return "", state, new_selected, status, *updates
        return gr.skip(), gr.skip(), gr.skip(), "Failed to create", *([gr.skip()] * (MAX_NOTEBOOKS * 2))
    except Exception as e:
        return gr.skip(), gr.skip(), gr.skip(), f"Error: {e}", *([gr.skip()] * (MAX_NOTEBOOKS * 2))


def _safe_rename(idx, new_name, state, selected_id, profile: gr.OAuthProfile | None):
    """Rename notebook at index."""
    try:
        if idx is None or idx < 0 or idx >= len(state):
            return gr.skip(), gr.skip(), gr.skip(), *([gr.skip()] * (MAX_NOTEBOOKS * 2))
        nb_id, _ = state[idx]
        name = (new_name or "").strip()
        if not name:
            return gr.skip(), gr.skip(), gr.skip(), "Enter a name.", *([gr.skip()] * (MAX_NOTEBOOKS * 2))
        user_id = _user_id(profile)
        if not user_id:
            return gr.skip(), gr.skip(), gr.skip(), "Please sign in", *([gr.skip()] * (MAX_NOTEBOOKS * 2))
        ok = rename_notebook(user_id, nb_id, name)
        if ok:
            notebooks = _get_notebooks(user_id)
            state = [(n["notebook_id"], n["name"]) for n in notebooks]
            updates = _build_row_updates(notebooks)
            return state, selected_id, f"Renamed to: {name}", *updates
        return gr.skip(), gr.skip(), gr.skip(), "Failed to rename", *([gr.skip()] * (MAX_NOTEBOOKS * 2))
    except Exception as e:
        return gr.skip(), gr.skip(), gr.skip(), f"Error: {e}", *([gr.skip()] * (MAX_NOTEBOOKS * 2))


def _safe_delete(idx, state, selected_id, profile: gr.OAuthProfile | None):
    """Delete notebook at index."""
    try:
        if idx is None or idx < 0 or idx >= len(state):
            return gr.skip(), gr.skip(), gr.skip(), *([gr.skip()] * (MAX_NOTEBOOKS * 2))
        nb_id, _ = state[idx]
        user_id = _user_id(profile)
        if not user_id:
            return gr.skip(), gr.skip(), gr.skip(), "Please sign in", *([gr.skip()] * (MAX_NOTEBOOKS * 2))
        ok = delete_notebook(user_id, nb_id)
        if ok:
            notebooks = _get_notebooks(user_id)
            state = [(n["notebook_id"], n["name"]) for n in notebooks]
            updates = _build_row_updates(notebooks)
            new_selected = notebooks[0]["notebook_id"] if notebooks else None
            return state, new_selected, "Notebook deleted", *updates
        return gr.skip(), gr.skip(), gr.skip(), "Failed to delete", *([gr.skip()] * (MAX_NOTEBOOKS * 2))
    except Exception as e:
        return gr.skip(), gr.skip(), gr.skip(), f"Error: {e}", *([gr.skip()] * (MAX_NOTEBOOKS * 2))


def _select_notebook(idx, state):
    """Set selected notebook when user interacts with a row."""
    if idx is None or idx < 0 or idx >= len(state):
        return gr.skip()
    return state[idx][0]


def _initial_load(profile: gr.OAuthProfile | None):
    """Load notebooks on app load. Uses HF OAuth profile for user_id."""
    user_id = _user_id(profile)
    notebooks = _get_notebooks(user_id)
    state = [(n["notebook_id"], n["name"]) for n in notebooks]
    selected = notebooks[0]["notebook_id"] if notebooks else None
    updates = _build_row_updates(notebooks)
    status = f"Signed in as {user_id}" if user_id else "Sign in with Hugging Face to manage notebooks."
    return state, selected, status, *updates


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


def _list_uploaded_pdfs(selected_id, profile: gr.OAuthProfile | None):
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


def _safe_remove_pdf(file_name, selected_id, profile: gr.OAuthProfile | None):
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


def _safe_ingest_url(url, selected_id, profile: gr.OAuthProfile | None):
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
    
def _safe_remove_url(url, selected_id, profile: gr.OAuthProfile | None):
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



def _build_row_updates(notebooks):
    """Return gr.update values for each row: visibility, then text value."""
    out = []
    for i in range(MAX_NOTEBOOKS):
        visible = i < len(notebooks)
        name = notebooks[i]["name"] if visible else ""
        out.append(gr.update(visible=visible))
        out.append(gr.update(value=name, visible=visible))
    return out

#Upload Handler Functions
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

# Quiz Handlers 
def _get_notebook_pdfs(notebook_id, profile: gr.OAuthProfile | None):
    user_id = _user_id(profile)
    if not user_id or not notebook_id:
        return gr.update(choices=[], value=None, visible=False)
    
    target_dir = Path("data") / "uploads" / user_id / str(notebook_id)
    if not target_dir.exists():
        return gr.update(choices=[], value=None, visible=False)
    
    pdfs = sorted([p.name for p in target_dir.glob("*.pdf")])
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

    if source_type_key == "pdf" and not pdf_source_id:
        return "Pick a PDF first.", [], *([gr.update(visible=False)] * 5 * 4), gr.update(visible=False), ""

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
                    # change this line for short_answer:
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

def _get_quiz_pdfs(source_type, notebook_id):
    if source_type != "PDF":
        return gr.update(visible=False, choices=[], value=None)
    if not notebook_id:
        return gr.update(visible=False, choices=[], value=None)
    
    # Search across all users for this notebook_id
    base = Path("data") / "uploads"
    pdfs = []
    if base.exists():
        for user_dir in base.iterdir():
            nb_dir = user_dir / str(notebook_id)
            if nb_dir.exists():
                pdfs = sorted([p.name for p in nb_dir.glob("*.pdf")])
                break
    
    print(f"DEBUG quiz pdfs found: {pdfs}")
    return gr.update(visible=True, choices=pdfs, value=pdfs[0] if pdfs else None)

def _quiz_pdf_dropdown_update(source_type, notebook_id, profile: gr.OAuthProfile | None):
    if source_type != "PDF":
        return gr.update(visible=False, choices=[], value=None)
    
    if not notebook_id:
        return gr.update(visible=True, choices=[], value=None)
    
    user_id = _user_id(profile)
    
    # Try with user_id first (production)
    if user_id:
        target_dir = Path("data") / "uploads" / user_id / str(notebook_id)
        if target_dir.exists():
            pdfs = sorted([p.name for p in target_dir.glob("*.pdf")])
            return gr.update(visible=True, choices=pdfs, value=pdfs[0] if pdfs else None)
    
    # Fallback for local dev (no OAuth): scan all user folders
    base = Path("data") / "uploads"
    if base.exists():
        for user_dir in base.iterdir():
            if not user_dir.is_dir():
                continue
            nb_dir = user_dir / str(notebook_id)
            if nb_dir.exists():
                pdfs = sorted([p.name for p in nb_dir.glob("*.pdf")])
                print(f"DEBUG (local fallback): notebook_id={notebook_id}, pdfs={pdfs}")
                return gr.update(visible=True, choices=pdfs, value=pdfs[0] if pdfs else None)
    
    return gr.update(visible=True, choices=[], value=None)

def _generate_btn_update(source_type, pdf_name):
    if source_type == "PDF":
        return gr.update(interactive=bool(pdf_name))
    return gr.update(interactive=True)

with gr.Blocks(
    title="NotebookLM Clone - Notebooks",
    theme=theme,
    css=CUSTOM_CSS,
) as demo:
    gr.HTML('<div class="container"><p class="hero">Notebook Manager</p><p class="sub">Create notebook below, then manage with Rename and Delete</p></div>')

    with gr.Row(elem_classes=["login-center"]):
        gr.Markdown("**Sign in with Hugging Face to access your notebooks**")
        with gr.Row(elem_classes=["login-btn-wrap"]):
            login_btn = gr.LoginButton(value="🤗 Login with Hugging Face", size="lg")

    nb_state = gr.State([])
    selected_notebook_id = gr.State(None)

    # Create section: text box + Create button
    with gr.Row():
        create_txt = gr.Textbox(
            label="Create notebook",
            placeholder="Enter new notebook name",
            value="",
            scale=3,
        )
        create_btn = gr.Button("Create", variant="primary", scale=1)

    with gr.Row():
        pdf_upload_btn = gr.UploadButton(
            "Upload PDFs",
            file_types=[".pdf"],
            file_count="multiple",
            type="filepath",
            variant="secondary",
        )

    with gr.Row():
        uploaded_pdf_dd = gr.Dropdown(
            label="Uploaded PDFs",
            choices=[],
            value=None,
            scale=3,
            allow_custom_value=False,
        )
        remove_pdf_btn = gr.Button("Remove selected PDF", variant="stop", scale=1)

    with gr.Row():
        url_txt = gr.Textbox(
            label="Ingest web URL",
            placeholder="https://example.com",
            value="",
            scale=3,
        )
        ingest_url_btn = gr.Button("Ingest URL", variant="primary", scale=1)
        remove_url_btn = gr.Button("Delete URL", variant="stop", scale=1)

    gr.Markdown("---")
    gr.Markdown("**Your notebooks** (selected notebook used for chat/ingestion)")

    # Rows: each notebook has [name] [Rename] [Delete]
    row_components = []
    row_outputs = []
    for i in range(MAX_NOTEBOOKS):
        with gr.Row(visible=False) as row:
            name_txt = gr.Textbox(
                value="",
                show_label=False,
                scale=3,
                min_width=200,
            )
            rename_btn = gr.Button("Rename", scale=1, min_width=80)
            delete_btn = gr.Button("Delete", variant="stop", scale=1, min_width=80)
            select_btn = gr.Button("Select", scale=1, min_width=70)
            row_components.append({"row": row, "name": name_txt, "rename": rename_btn, "delete": delete_btn, "select": select_btn})
            row_outputs.extend([row, name_txt])

    status = gr.Markdown("Sign in with Hugging Face to manage notebooks.", elem_classes=["status"])

    # Create button
    create_btn.click(
        _safe_create,
        inputs=[create_txt, nb_state, selected_notebook_id],
        outputs=[create_txt, nb_state, selected_notebook_id, status] + row_outputs,
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

   # Quiz Section 
    gr.Markdown("---")
    gr.Markdown("## Generate Quiz")

    quiz_source_type = gr.Radio(
        choices=["Text", "PDF", "URL", "All"],
        value="All",
        label="Source type",
    )


    quiz_pdf_dd = gr.Dropdown(
        label="Select PDF (select a notebook first if empty)",
        choices=[],
        value=None,
        visible=False,
    )

    demo.load(_initial_load, inputs=None, outputs=[nb_state, selected_notebook_id, status] + row_outputs, api_name=False)
    demo.load(_list_uploaded_pdfs, inputs=[selected_notebook_id], outputs=[uploaded_pdf_dd], api_name=False)
    demo.load(
        _quiz_pdf_dropdown_update,
        inputs=[quiz_source_type, selected_notebook_id],
        outputs=[quiz_pdf_dd],
        api_name=False,
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

    for i in range(MAX_NOTEBOOKS):
        select_btn = row_components[i]["select"]
        def _on_select(i=i):
            return "Selected notebook updated. Use this for chat/ingestion."
        
        select_btn.click(
            _select_notebook,
            inputs=[gr.State(i), nb_state],
            outputs=[selected_notebook_id],
            api_name=False,
        ).then(
            _on_select, None, [status]
        ).then(
            _list_uploaded_pdfs, inputs=[selected_notebook_id], outputs=[uploaded_pdf_dd]
        ).then(
            _quiz_pdf_dropdown_update,
            inputs=[quiz_source_type, selected_notebook_id],
            outputs=[quiz_pdf_dd],
            api_name=False,
        ).then(
            _generate_btn_update,
            inputs=[quiz_source_type, quiz_pdf_dd],
            outputs=[generate_quiz_btn],
            api_name=False,
        )
        

    submit_btn.click(
        _do_upload,
        inputs=[txt_input, txt_title, selected_notebook_id],
        outputs=[upload_status],
    )

    quiz_source_type.change(
        _quiz_pdf_dropdown_update,
        inputs=[quiz_source_type, selected_notebook_id],
        outputs=[quiz_pdf_dd],
        api_name=False,
    ).then(
        _generate_btn_update,
        inputs=[quiz_source_type, quiz_pdf_dd],
        outputs=[generate_quiz_btn],
        api_name=False,
    )

    quiz_pdf_dd.change(
        _generate_btn_update,
        inputs=[quiz_source_type, quiz_pdf_dd],
        outputs=[generate_quiz_btn],
        api_name=False,
    )

   

    quiz_all_outputs = [quiz_status, quiz_state]
    for c in quiz_components:
        quiz_all_outputs += [c["group"], c["text"], c["radio"], c["textbox"]]
    quiz_all_outputs += [submit_quiz_btn, quiz_results]

    generate_quiz_btn.click(
    lambda: gr.update(value="Generating quiz..."),
    inputs=[],
    outputs=[quiz_status],
    api_name=False,
    ).then(
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

demo.launch()



