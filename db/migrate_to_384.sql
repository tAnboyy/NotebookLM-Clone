-- Migration: Switch from 1536-dim to 384-dim embeddings (MiniLM)
-- Run this in Supabase SQL Editor if you already have the chunks table with vector(1536)

-- 1. Drop the ivfflat index (required before altering column)
drop index if exists idx_chunks_embedding;

-- 2. Clear existing chunks (old 1536-dim embeddings are incompatible)
truncate table chunks;

-- 3. Replace embedding column with 384-dim version
alter table chunks drop column embedding;
alter table chunks add column embedding vector(384);

-- 4. Recreate the ivfflat index (run AFTER ingesting new PDF/TXT - requires rows)
-- create index if not exists idx_chunks_embedding on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- 5. Update match_chunks RPC
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
