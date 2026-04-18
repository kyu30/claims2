---
title: Claims2
sdk: static
app_file: index.html
---

# Claim Mapping MVP UI

Minimal UI to map article **paragraphs** to subclaims and superclaims using four JSON data files plus an optional offline collapse artifact.

## Data files (required)

Place these next to `index.html` (or serve the folder over HTTP):

- `greenwashing_claim_history.json` — `claims` object: per subclaim, `current_text` and `history[].source_snippet`
- `greenwashing_codebook.json` — object `{ "NC_*": "subclaim text", ... }` (canonical subclaim text; falls back to `current_text` in claim history if missing)
- `greenwashing_superclaims.json` — object `{ "SC_*": "superclaim text", ... }`
- `claim_superclaim_map.json` — subclaim → superclaim mapping: either a plain object `{ "NC_*": "SC_*", ... }`, an array of `{ subclaim_id, superclaim_id }` (optional `superclaim_text`), or a combined object keyed by subclaim id with `superclaim_id` / `superclaim_text` fields (see `runs/validation_run_20260306/claims_validation.py` for supported shapes)

## Optional: collapse artifact

- `subclaim_bertopic_collapse.json` — generated offline; adds BERTopic / DBSCAN **collapse hints** and **cosine similarity** (`hierarchy_confidence`) between each subclaim’s text and its mapped superclaim text (same embedding space as the build script). If missing, the UI still runs.

## Usage

- Start a simple static server from this folder (for example: `python -m http.server 8000`).
- Paste an article into the text area and click **Analyze paragraphs**.

For each detected sentence, the UI shows:

- The **subclaim text** with its `NC_*` id
- The **superclaim text** with its `SC_*` id
- **Collapse hints** from the BERTopic artifact when present (no live classifier API)

If no close match is found for a sentence, the UI shows a “No mapping found” message for that row.

## Offline artifact (BERTopic / DBSCAN)

Paragraph→subclaim matching uses the **local** overlap / LCS heuristic in the browser. The **superclaim** column shows **precomputed** collapse metadata and **subclaim↔superclaim cosine similarity** from `subclaim_bertopic_collapse.json`.

### Build

```bash
pip install -r requirements.txt
python scripts/build_subclaim_collapse_bertopic.py
```

This reads the four JSON files above (for bundle fingerprinting and mappings). It clusters subclaims using text from the **codebook** when present, otherwise `current_text` in claim history (only subclaims that appear in `claim_superclaim_map.json`). It then encodes each subclaim with its mapped **superclaim** text and writes **cosine similarity** per row. Output: `subclaim_bertopic_collapse.json` with `claims_bundle_version`.

- **Default** (`--embedding-backend auto`): uses `sentence-transformers` when installed; otherwise **TF‑IDF + SVD** (no PyTorch).
- **Full BERTopic** (topic labels): install `bertopic` and use `--embedding-backend sentence_transformers --cluster-backend bertopic` (needs a Python where `hdbscan` wheels install, e.g. 3.10+ on Windows).

Example without deep-learning stacks:

```bash
python scripts/build_subclaim_collapse_bertopic.py --embedding-backend tfidf --cluster-backend sklearn
```

### Claim bundle versioning

- **`claims_bundle_version`** in the artifact is a short hash over the four canonical JSON files. When any of them change, regenerate the artifact so collapse metadata stays aligned.
- Optionally set **`claims_version`** in the root of `greenwashing_claim_history.json`; the build script copies it into the artifact for traceability.
- After analysis, the status line shows the artifact id (`artifact …`) when a collapse file is loaded.

## Optional backend (LLM)

The static UI does **not** require the backend. The FastAPI app under `backend/` is optional if you want a separate LLM-based `/confidence` endpoint for other tools.
