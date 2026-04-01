# Claim Mapping MVP UI

Minimal UI to map article sentences to subclaims and superclaims derived from `greenwashing_claim_history.json`.

## Usage

- Place `index.html`, `styles.css`, `script.js`, `greenwashing_claim_history.json`, mapping JSONL files under `8B model/`, and the generated artifact `subclaim_bertopic_collapse.json` in the same folder.
- Start a simple static server from this folder (for example: `python -m http.server 8000`).
- Paste an article into the text area and click **Analyze sentences**.

For each detected sentence, the UI shows:

- The **subclaim text** (from claim history) with an ID in parentheses (`NC_*`).
- The corresponding **superclaim text** with its `SC_*` ID.
- **Offline** signals from the BERTopic artifact (no live classifier API):
  - **Semantic confidence**: cosine similarity between `current_text` and the mapped superclaim text (computed when you build the artifact).
  - **Collapse hints**: whether other subclaims share the same BERTopic topic cluster (candidates to merge).

If no close match is found for a sentence, the UI displays a "No mapping found" message for that row.

## Offline artifact (BERTopic + embeddings)

Mapping selection still uses the **local** heuristic in the browser. The **superclaim** column shows **precomputed** scores and flags from `subclaim_bertopic_collapse.json` — no OpenAI or other live classification API at click time.

### Build

```bash
pip install -r requirements.txt
python scripts/build_subclaim_collapse_bertopic.py
```

This reads `greenwashing_claim_history.json`, `8B model/validated_mappings.jsonl`, and `8B model/flagged_mappings.jsonl`, clusters subclaim `current_text` embeddings, and writes `subclaim_bertopic_collapse.json` with a `claims_bundle_version` fingerprint.

- **Default** (`--embedding-backend auto`): uses `sentence-transformers` when installed; otherwise **TF‑IDF + SVD** (no PyTorch).
- **Full BERTopic** (topic labels): install `bertopic` and use `--embedding-backend sentence_transformers --cluster-backend bertopic` (needs a Python where `hdbscan` wheels install, e.g. 3.10+ on Windows).

Example without deep-learning stacks:

```bash
python scripts/build_subclaim_collapse_bertopic.py --embedding-backend tfidf --cluster-backend sklearn
```

### Claim bundle versioning (for future API-fed claims)

- **`claims_bundle_version`** in the artifact is a short hash over the **claims file** and **mapping JSONL** contents. When claims or mappings change (including when you load a new snapshot from an API), regenerate the artifact so the UI stays aligned with the data.
- Optionally set **`claims_version`** in the root of `greenwashing_claim_history.json`; the build script copies it into the artifact for traceability.
- After analysis, the status line shows the artifact id (`artifact …`) so you can confirm which build is in use.

## Optional backend (LLM)

The static UI does **not** require the backend for confidence. The FastAPI app under `backend/` is optional if you still want a separate LLM-based `/confidence` endpoint for other tools.
