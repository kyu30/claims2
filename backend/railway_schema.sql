-- Railway Postgres schema for proposal persistence (Option 2).
-- Run this against your Railway Postgres (Railway dashboard → Postgres → Connect → DATABASE_URL).
--
-- Hugging Face Docker Space: add DATABASE_URL as a Space secret (same URL Railway shows).
-- Use a URL with SSL if required, e.g. append ?sslmode=require when the provider expects TLS.
--
-- This app uses DATABASE_URL for taxonomy_proposals / taxonomy_merge_log when set; Supabase
-- client variables are then unnecessary for those tables. Claim JSON files still read from
-- disk unless you use Supabase Storage or set CLAIMS_JSON_DIR to a writable directory.

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

