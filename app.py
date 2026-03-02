from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (parent of NotebookLM-Clone) so HF_TOKEN etc. are available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

from datetime import datetime
import gradio as gr

from backend.notebook_service import create_notebook, list_notebooks, rename_notebook, delete_notebook

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


def _build_row_updates(notebooks):
    """Return gr.update values for each row: visibility, then text value."""
    out = []
    for i in range(MAX_NOTEBOOKS):
        visible = i < len(notebooks)
        name = notebooks[i]["name"] if visible else ""
        out.append(gr.update(visible=visible))
        out.append(gr.update(value=name, visible=visible))
    return out

# ── Upload Handler Functions ──────────────────────────────────
def _do_upload(text_content, title, notebook_id, profile: gr.OAuthProfile | None):
    """Handle direct text input and ingestion."""
    from backend.ingestion_txt import ingest_txt, list_sources

    user_id = _user_id(profile)

    if not user_id:
        return "❌ Please sign in first.", ""
    if not notebook_id:
        return "❌ Please select a notebook first.", ""
    if not text_content or not text_content.strip():
        return "❌ No text entered.", ""

    try:
        # Use title as filename, fallback to timestamp
        filename = (title or "").strip()
        if not filename:
            filename = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not filename.endswith(".txt"):
            filename = filename + ".txt"

        # Convert text to bytes for ingestion pipeline
        file_bytes = text_content.encode("utf-8")

        result = ingest_txt(
            file_bytes=file_bytes,
            filename=filename,
            notebook_id=notebook_id,
            user_id=user_id
        )

        meta = result["metadata"]
        status_msg = (
            f"✅ **{result['filename']}** saved successfully!\n\n"
            f"- Size: {meta['size_bytes'] / 1024:.1f} KB"
        )

        #sources = list_sources(notebook_id)
        return status_msg, ""

    except ValueError as e:
        return f"❌ {str(e)}", ""
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}", ""

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

    demo.load(_initial_load, inputs=None, outputs=[nb_state, selected_notebook_id, status] + row_outputs)

    # Create button
    create_btn.click(
        _safe_create,
        inputs=[create_txt, nb_state, selected_notebook_id],
        outputs=[create_txt, nb_state, selected_notebook_id, status] + row_outputs,
    )

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
        )
        delete_btn.click(
            _safe_delete,
            inputs=[gr.State(i), nb_state, selected_notebook_id],
            outputs=[nb_state, selected_notebook_id, status] + row_outputs,
        )
        def _on_select():
            return "Selected notebook updated. Use this for chat/ingestion."
        select_btn.click(
            _select_notebook,
            inputs=[gr.State(i), nb_state],
            outputs=[selected_notebook_id],
        ).then(_on_select, None, [status])

    # ── Text Input Section ────────────────────────────────────
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
    sources_display = gr.Markdown("")

    submit_btn.click(
        _do_upload,
        inputs=[txt_input, txt_title, selected_notebook_id],
        outputs=[upload_status, sources_display],
    )

    selected_notebook_id.change(
        _load_sources,
        inputs=[selected_notebook_id],
        outputs=[sources_display],
    )

demo.launch()
