-- Railway Postgres schema for proposal persistence (Option 2).
-- Run this against your Railway database before deploying the Space.

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

