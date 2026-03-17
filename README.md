# Claim Mapping MVP UI

Minimal UI to map article sentences to subclaims and superclaims derived from `greenwashing_claim_history.json`.

## Usage

- Place `index.html`, `styles.css`, `script.js` and `greenwashing_claim_history.json` in the same folder (they already are in this workspace).
- Start a simple static server from this folder (for example: `python -m http.server 8000`).
- Start the backend (for the live LLM confidence score).
- Paste an article into the text area and click **Analyze sentences**.

For each detected sentence, the UI shows:

- The **subclaim text** (taken from `source_snippet`) with an ID in parentheses (`NC_*` only).
- The corresponding **superclaim text** (the normalized claim `current_text`) with its `SC_*` ID in parentheses.

If no close match is found for a sentence, the UI displays a "No mapping found" message for that row.

## Backend (live confidence scoring)

The **mapping selection** remains the same as the MVP (local heuristic match), but the **confidence score shown in the UI** is computed live by `gpt-5-mini` via the backend.

### Setup

- Ensure `.env` contains:
  - `OPENAI_API_KEY=...`
  - `OPENAI_MODEL=gpt-5-mini` (optional; defaults to `gpt-5-mini`)

### Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start backend on port 8001:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8001
```

Start static UI on port 8000:

```bash
python -m http.server 8000
```

