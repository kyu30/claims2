"""
Offline pipeline: compute superclaim↔superclaim similarity and write a small
static artifact consumed by the UI.

The UI uses this to show a chip on each mapped superclaim indicating whether
that superclaim is semantically similar to any other superclaim.

Usage (from repo root):
  pip install -r requirements.txt
  python scripts/build_superclaim_similarity.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPERCLAIMS = ROOT / "greenwashing_superclaims.json"
DEFAULT_OUT = ROOT / "superclaim_similarity.json"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_sc(raw: str) -> str:
    s = str(raw).strip()
    if not s:
        return ""
    if s.startswith("SC_"):
        return s
    body = s.removeprefix("SC_").removeprefix("NC_")
    return f"SC_{body}"


def _load_superclaims(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object of {{id: text}}")
    out: dict[str, str] = {}
    for k, v in data.items():
        sid = _normalize_sc(k)
        text = str(v if v is not None else "").strip()
        if sid and text:
            out[sid] = text
    return out


def _fit_tfidf_embeddings(
    docs: list[str],
    *,
    max_features: int = 40000,
) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    min_df = 1 if len(docs) < 120 else 2
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=0.92,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs)
    # With TF-IDF, cosine similarity is dot product after L2 normalize.
    from sklearn.preprocessing import normalize

    X = normalize(X, norm="l2", copy=False)
    return X


def _topk_similar(
    X: Any,
    ids: list[str],
    *,
    top_k: int,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    # X is CSR TF-IDF normalized. For each row, compute sparse cosine sims by dot with X.T.
    n = len(ids)
    out: dict[str, dict[str, Any]] = {}
    XT = X.T
    for i, scid in enumerate(ids):
        sims_sparse = X[i].dot(XT)  # 1 x n sparse
        # Convert to dense row (small n); keep simple and deterministic.
        sims = np.asarray(sims_sparse.toarray()).ravel()
        if sims.shape[0] != n:
            raise RuntimeError("unexpected similarity shape")
        sims[i] = -1.0  # exclude self

        # candidates above threshold
        cand_idx = np.where(sims >= threshold)[0]
        if cand_idx.size:
            # sort candidates by similarity desc then id
            cand = sorted(
                ((int(j), float(sims[int(j)])) for j in cand_idx),
                key=lambda t: (-t[1], ids[t[0]]),
            )
            cand = cand[: max(1, top_k)]
            similar_with = [
                {"id": ids[j], "similarity": round(sim, 4)} for j, sim in cand
            ]
            max_sim = float(cand[0][1]) if cand else 0.0
            out[scid] = {
                "max_similarity": round(max_sim, 4),
                "similar_flag": True,
                "similar_with": similar_with,
            }
        else:
            # still compute max similarity for debugging/inspection
            max_sim = float(np.max(sims)) if n > 1 else 0.0
            max_sim = 0.0 if not np.isfinite(max_sim) else max_sim
            out[scid] = {
                "max_similarity": round(max(0.0, max_sim), 4),
                "similar_flag": False,
                "similar_with": [],
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build superclaim similarity artifact (TF-IDF cosine).",
    )
    parser.add_argument("--superclaims-json", type=Path, default=DEFAULT_SUPERCLAIMS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("Missing scikit-learn. Install with: pip install scikit-learn numpy")
        return 1

    p = args.superclaims_json
    if not p.is_file():
        print(f"Superclaims JSON not found: {p}")
        return 1

    superclaims = _load_superclaims(p)
    if len(superclaims) < 2:
        print("Need at least 2 superclaims with non-empty text.")
        return 1

    ids = sorted(superclaims.keys())
    docs = [superclaims[i] for i in ids]
    X = _fit_tfidf_embeddings(docs)
    rows = _topk_similar(
        X,
        ids,
        top_k=max(1, int(args.top_k)),
        threshold=float(args.threshold),
    )

    out_obj: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": p.name,
        "source_sha256": _file_sha256(p),
        "similarity_backend": "tfidf_cosine",
        "threshold": float(args.threshold),
        "top_k": int(args.top_k),
        "superclaims": rows,
    }

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    flagged = sum(1 for v in rows.values() if v.get("similar_flag"))
    print(f"Wrote {out_path} (similar_flag=true for {flagged}/{len(rows)} superclaims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

