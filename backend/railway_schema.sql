-- Railway Postgres schema for proposal persistence (Option 2).
-- Run this against your Railway Postgres (Railway dashboard → Postgres → Connect → DATABASE_URL).
--
-- Hugging Face Docker Space: add DATABASE_URL as a Space secret (same URL Railway shows).
-- Use a URL with SSL if required, e.g. append ?sslmode=require when the provider expects TLS.
--
-- This app uses DATABASE_URL for taxonomy_proposals / taxonomy_merge_log when set.
-- Optional: set env POSTGRES_TAXONOMY_TABLES=1 to read/write taxonomy_superclaims and
-- taxonomy_subclaims from this database (same shape as legacy Supabase tables).
-- Claim history still uses greenwashing_claim_history.json (disk or Supabase Storage).

create table if not exists taxonomy_proposals (
  id text primary key,
  type text not null,
  status text not null,
  created_at timestamptz not null,
  bundle_version text not null,
  paragraph text not null,
  payload jsonb not null,
  rationale text not null default '',
  reviewed_by text,
  reviewed_at timestamptz,
  applied_by text,
  applied_at timestamptz
);

create index if not exists taxonomy_proposals_status_created_at
  on taxonomy_proposals (status, created_at desc);

create table if not exists taxonomy_merge_log (
  id bigserial primary key,
  proposal_id text not null,
  merge_type text not null,
  bundle_version text not null,
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists taxonomy_superclaims (
  id text primary key,
  text text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists taxonomy_subclaims (
  id text primary key,
  text text not null,
  superclaim_id text not null references taxonomy_superclaims(id) on update cascade on delete restrict,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists taxonomy_subclaims_superclaim_idx
  on taxonomy_subclaims (superclaim_id);

