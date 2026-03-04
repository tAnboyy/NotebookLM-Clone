---
title: NotebookLM Clone
emoji: 📓
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
hf_oauth_scopes:
  - email
  - read-repos
---

# NotebookLM Clone

A document-based RAG (Retrieval-Augmented Generation) application inspired by Google NotebookLM. Upload PDFs, ingest web URLs, add text, then chat with your sources—with grounded answers and citations. Generate reports, quizzes, and podcast transcripts/audio.

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/rahulrb99/notebooklm-test)  
**Repository:** [GitHub](https://github.com/tAnboyy/NotebookLM-Clone)

---

## Features

- **Notebook Management** — Create, rename, delete notebooks; select one for chat and ingestion
- **Multi-Source Ingestion** — PDF uploads, single-page URL scraping, direct text input
- **RAG Chat** — Ask questions; answers are grounded in retrieved chunks with `[1]` `[2]` citations
- **Artifact Generation** — Reports, quizzes (with answer key), podcast transcript and audio
- **Authentication** — Hugging Face OAuth; per-user data isolation
- **Storage** — Supabase (PostgreSQL + pgvector + Storage); per-user/per-notebook structure

---

## Prerequisites

- **Python 3.10, 3.11, or 3.12** (Gradio has issues with Python 3.13)
- **Supabase** project with pgvector
- **Hugging Face** account and token (for OAuth and LLM)

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/tAnboyy/NotebookLM-Clone.git
cd NotebookLM-Clone
```

### 2. Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and create a project
2. In **SQL Editor**, run the schema:

```bash
# Copy contents of db/schema.sql and execute in Supabase SQL Editor
```

Schema creates: `notebooks`, `messages`, `artifacts`, `chunks` (with `vector(384)`), `sources`, and the `match_chunks` RPC for vector search.

### 3. Configure Storage

1. In Supabase **Storage**, create a bucket named `notebooklm` (or set `SUPABASE_BUCKET`)
2. Enable public read if needed for artifact URLs

### 4. Environment Variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
HF_TOKEN=your-huggingface-token
```

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Service role key (not anon key) |
| `HF_TOKEN` | Hugging Face token (Settings → Access Tokens) |
| `LLM_MODEL` | (Optional) Default: `meta-llama/Llama-3.2-3B-Instruct:together` |
| `SUPABASE_BUCKET` | (Optional) Storage bucket name; default: `notebooklm` |

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the App

**Windows (recommended):**

```powershell
.\run.bat
```

**Manual:**

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

---

## Hugging Face Space Deployment

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Add **Secrets**: `SUPABASE_URL`, `SUPABASE_KEY`, `HF_TOKEN`
3. Push to the Space:

```bash
git remote add huggingface https://hf.co/spaces/YOUR_USERNAME/YOUR_SPACE
git push huggingface main
```

OAuth works automatically on HF Spaces.

---

## Architecture

### High-Level Data Flow

```
User → Upload PDF/URL/Text → Ingestion → Chunk + Embed → chunks table (pgvector)
User → Query → Embed query → match_chunks RPC → Top-k chunks → LLM → Answer + citations
```

### Module Responsibilities

| Module | Responsibility |
|--------|-----------------|
| `embedding_service.py` | Shared MiniLM embeddings (384-dim) for documents and queries |
| `ingestion_service.py` | PDF and URL extraction, chunking, embedding, storage |
| `ingestion_txt.py` | Text file ingestion, chunking, embedding, sources table |
| `retrieval_service.py` | Vector similarity search via `match_chunks` RPC |
| `rag_service.py` | RAG orchestration: retrieve → prompt → LLM → validate citations |
| `report_service.py` | Report generation from chunks (scope: all/pdf/url/text) |
| `quiz_service.py` | Quiz generation from chunks (multiple choice, true/false, short answer) |
| `podcast_service.py` | Podcast script and audio generation |
| `llm_client.py` | OpenAI-compatible client for HF router (Together, Groq, etc.) |
| `chat_service.py` | Persist and load chat messages per notebook |
| `storage.py` | Supabase Storage abstraction (artifacts, sources) |

### Storage Structure

```
Supabase:
  notebooks     — user_id, name
  chunks        — notebook_id, content, embedding (vector 384), metadata
  messages      — notebook_id, role, content (chat history)
  artifacts     — notebook_id, type, storage_path
  sources       — notebook_id, user_id, filename, status

Storage bucket:
  {user_id}/{notebook_id}/  — PDFs, artifacts, sources
```

---

## RAG Techniques

This project implements **Basic RAG** (Naive RAG):

- Fixed-size chunking (1200 chars / 200 overlap for PDF/URL; 400 words / 40 overlap for text)
- Dense retrieval via cosine similarity on MiniLM embeddings
- Single-stage retrieval (no reranking)
- Direct context injection with citation instructions

See `backend/report.docx` for a detailed comparison of RAG techniques (Basic RAG, HyDE, Reranking, Semantic Chunking) and performance tradeoffs. Reference: [NirDiamant RAG Techniques](https://github.com/NirDiamant/RAG_Techniques).

---

## Project Structure

```
clone_v2/
├── app.py                 # Gradio UI entry point
├── backend/
│   ├── embedding_service.py
│   ├── ingestion_service.py
│   ├── ingestion_txt.py
│   ├── retrieval_service.py
│   ├── rag_service.py
│   ├── report_service.py
│   ├── quiz_service.py
│   ├── podcast_service.py
│   ├── llm_client.py
│   ├── chat_service.py
│   ├── notebook_service.py
│   ├── artifacts_service.py
│   ├── storage.py
│   └── db.py
├── db/
│   ├── schema.sql
│   └── migrate_to_384.sql
├── requirements.txt
├── run.bat
└── README.md
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: openai` | `pip install openai` |
| Gradio fails on Python 3.13 | Use Python 3.10–3.12; run `.\run.bat` |
| `SUPABASE_URL and SUPABASE_KEY must be set` | Add `.env` with correct values |
| `HF_TOKEN environment variable is required` | Add `HF_TOKEN` to `.env` |
| OAuth not working locally | HF OAuth works fully on Spaces; locally it mocks with your HF profile |

---

## License

See repository for license details.
