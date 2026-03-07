# 📓 NotebookLM Clone

A RAG-based document chat application. Upload PDFs, ingest URLs, or paste text—then chat with your sources, generate reports, quizzes, and podcasts.

**🚀 Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/rahulrb99/notebooklm-test)

**📂 Original Repository:** [tAnboyy/NotebookLM-Clone](https://github.com/tAnboyy/NotebookLM-Clone)

---

## Features

- **Chat with documents** – Ask questions about your uploaded PDFs, URLs, and text
- **Citation display** – See source chunks and excerpts for each answer
- **Report generation** – Summarize content across all or selected sources
- **Quiz generation** – Auto-generate multiple choice, true/false, and short-answer questions
- **Podcast generation** – Create podcast scripts and audio from notebook content
- **Multi-user** – Hugging Face OAuth; chat and notebooks isolated per user

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Gradio 4.44 |
| LLM | Hugging Face Inference (router.huggingface.co) |
| Embeddings | Sentence Transformers (MiniLM / BGE) |
| Vector DB | Supabase (pgvector) |
| RAG | Semantic chunking, cross-encoder reranking |

---

## My Contributions

- **Citation display** – Accordion showing source chunks and metadata for each RAG answer
- **Retrieval improvements** – Cross-encoder reranking, semantic chunking, similarity threshold
- **Chat isolation** – Notebook ownership checks so users only see their own chat history
- **Quiz service fix** – Migrated from deprecated `api-inference.huggingface.co` to `router.huggingface.co`
- **CI/CD** – Deployed to Hugging Face Spaces with automated builds

---

## Quick Start

### Prerequisites

- Python 3.10, 3.11, or 3.12 (Gradio does not work reliably with 3.13)
- [Supabase](https://supabase.com) project
- [Hugging Face](https://huggingface.co) token

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/rahulrb99/NotebookLM-Clone.git
   cd NotebookLM-Clone
   ```

2. **Create a `.env` file** in the project root:

   ```
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_service_role_key
   HF_TOKEN=your_huggingface_token
   ```

3. **Run the database schema** in Supabase SQL Editor (see `db/schema.sql`)

4. **Install dependencies and run**

   ```powershell
   pip install -r requirements.txt
   python app.py
   ```

   Or on Windows:

   ```powershell
   .\run.bat
   ```

5. Open **http://127.0.0.1:7860** in your browser

---

## Project Structure

```
├── app.py                 # Gradio UI
├── backend/
│   ├── chat_service.py    # Chat persistence + ownership validation
│   ├── chunking.py        # Semantic chunking for RAG
│   ├── embedding_service.py
│   ├── ingestion_service.py   # PDF, URL ingestion
│   ├── ingestion_txt.py    # Text ingestion
│   ├── llm_client.py      # LLM via HF router
│   ├── podcast_service.py
│   ├── quiz_service.py
│   ├── rag_service.py     # RAG chat + citations
│   ├── retrieval_service.py   # Vector search + reranking
│   └── ...
├── db/
│   └── schema.sql         # Supabase schema
└── requirements.txt
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service role key |
| `HF_TOKEN` | Hugging Face token (LLM, OAuth) |
| `LLM_MODEL` | Optional; default uses Together/HF |
| `EMBEDDING_MODEL` | Optional; default: all-MiniLM-L6-v2 |
| `USE_RERANKER` | Optional; default: true |

---

## License

See the [original repository](https://github.com/tAnboyy/NotebookLM-Clone) for license information.
