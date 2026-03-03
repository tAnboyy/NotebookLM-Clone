-- Run this in Supabase SQL Editor

-- notebooks (existing)
create table if not exists notebooks (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  name varchar(255) not null default 'Untitled Notebook',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);
create index if not exists idx_notebooks_user_id on notebooks(user_id);

-- messages
create table if not exists messages (
  id uuid primary key default gen_random_uuid(),
  notebook_id uuid not null references notebooks(id) on delete cascade,
  role text not null,
  content text not null,
  created_at timestamptz default now()
);
create index if not exists idx_messages_notebook_id on messages(notebook_id);

-- artifacts
create table if not exists artifacts (
  id uuid primary key default gen_random_uuid(),
  notebook_id uuid not null references notebooks(id) on delete cascade,
  type text not null,
  storage_path text not null,
  created_at timestamptz default now()
);
create index if not exists idx_artifacts_notebook_id on artifacts(notebook_id);

-- pgvector extension for embeddings
create extension if not exists vector;

-- chunks with embeddings (for RAG) - 384 dims for MiniLM
create table if not exists chunks (
  id uuid primary key default gen_random_uuid(),
  notebook_id uuid not null references notebooks(id) on delete cascade,
  source_id text,
  content text not null,
  embedding vector(384),
  metadata jsonb,
  created_at timestamptz default now()
);
create index if not exists idx_chunks_notebook_id on chunks(notebook_id);

-- Vector index for fast similarity search (run after chunks have data; ivfflat requires rows)
create index if not exists idx_chunks_embedding on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- RPC for RAG retrieval: top-k chunks by cosine similarity, filtered by notebook_id
create or replace function match_chunks(
  query_embedding vector(384),
  match_count int,
  p_notebook_id uuid
)
returns table (id uuid, content text, metadata jsonb, similarity float)
language plpgsql as $$
begin
  return query
  select c.id, c.content, c.metadata,
         1 - (c.embedding <=> query_embedding) as similarity
  from chunks c
  where c.notebook_id = p_notebook_id
    and c.embedding is not null
  order by c.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- sources table (ingestion pipeline)
create table if not exists sources (
    id uuid primary key default gen_random_uuid(),
    notebook_id uuid not null references notebooks(id) on delete cascade,
    user_id text not null,
    filename text not null,
    file_type text not null,
    status text not null default 'PENDING',
    storage_path text,
    extracted_text text,
    metadata jsonb default '{}',
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);
create index if not exists idx_sources_notebook_id on sources(notebook_id);
create index if not exists idx_sources_user_id on sources(user_id);
create index if not exists idx_sources_status on sources(status);