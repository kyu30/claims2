"""
Offline pipeline: BERTopic (or sklearn DBSCAN) clusters subclaim text, then adds
cosine similarity (subclaim vs mapped superclaim) using the same embedding space.

Subclaim text prefers ``greenwashing_codebook.json``, then ``current_text`` in
``greenwashing_claim_history.json``. Superclaim text comes from
``greenwashing_superclaims.json`` via ``claim_superclaim_map.json``.

Writes subclaim_bertopic_collapse.json (bundle fingerprint + per-subclaim rows:
topic / collapse hints + hierarchy_confidence).

No live classification APIs — output is consumed as static JSON by the UI.

Usage (from repo root):
  pip install -r requirements.txt
  python scripts/build_subclaim_collapse_bertopic.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLAIMS = ROOT / "greenwashing_claim_history.json"
DEFAULT_MAP = ROOT / "claim_superclaim_map.json"
DEFAULT_SUPERCLAIMS = ROOT / "greenwashing_superclaims.json"
DEFAULT_CODEBOOK = ROOT / "greenwashing_codebook.json"
DEFAULT_OUT = ROOT / "subclaim_bertopic_collapse.json"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_dual_prefix(s: str, a: str, b: str) -> str:
    if s.startswith(a):
        return s[len(a) :]
    if s.startswith(b):
        return s[len(b) :]
    return s


def _normalize_nc(raw: str) -> str:
    s = str(raw).strip()
    if not s:
        return ""
    if s.startswith("NC_"):
        return s
    body = _strip_dual_prefix(s, "NC_", "SC_")
    return f"NC_{body}"


def _normalize_sc(raw: str) -> str:
    s = str(raw).strip()
    if not s:
        return ""
    if s.startswith("SC_"):
        return s
    body = _strip_dual_prefix(s, "SC_", "NC_")
    return f"SC_{body}"


def _load_id_text_json(path: Path, kind: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object of {{id: text}}")
    out: dict[str, str] = {}
    for k, v in data.items():
        text = str(v if v is not None else "").strip()
        nid = _normalize_nc(str(k)) if kind == "subclaim" else _normalize_sc(str(k))
        if nid:
            out[nid] = text
    return out


def _parse_claim_superclaim_map(obj: Any) -> dict[str, str]:
    """Normalized subclaim_id -> superclaim_id."""
    pairs: list[tuple[str, str]] = []
    if obj is None:
        return {}
    if isinstance(obj, dict):
        keys = list(obj.keys())
        first_key = keys[0] if keys else None
        sample = obj[first_key] if first_key is not None else None
        is_combined = (
            sample is not None
            and isinstance(sample, dict)
            and not isinstance(sample, list)
            and (
                "superclaim_id" in sample
                or "superclaimId" in sample
                or "sc_id" in sample
            )
        )
        if is_combined:
            for sub_id, record in obj.items():
                if not isinstance(record, dict):
                    continue
                sc = (
                    record.get("superclaim_id")
                    or record.get("superclaimId")
                    or record.get("sc_id")
                    or record.get("SC")
                )
                if sc is None:
                    continue
                pairs.append((_normalize_nc(str(sub_id)), _normalize_sc(str(sc))))
        else:
            for nc, sc in obj.items():
                pairs.append((_normalize_nc(str(nc)), _normalize_sc(str(sc))))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((_normalize_nc(str(item[0])), _normalize_sc(str(item[1]))))
            elif isinstance(item, dict):
                nc = (
                    item.get("subclaim_id")
                    or item.get("nc_id")
                    or item.get("subclaim")
                    or item.get("NC")
                )
                sc = (
                    item.get("superclaim_id")
                    or item.get("sc_id")
                    or item.get("superclaim")
                    or item.get("SC")
                )
                if nc is None or sc is None:
                    continue
                pairs.append((_normalize_nc(str(nc)), _normalize_sc(str(sc))))
    out: dict[str, str] = {}
    for sid, scid in pairs:
        if sid and scid:
            out[sid] = scid
    return out


def _encode_subclaims(
    docs: list[str],
    embedding_model: Any,
    *,
    verbose: bool = True,
) -> np.ndarray:
    return embedding_model.encode(
        docs,
        batch_size=64,
        show_progress_bar=verbose,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def fit_tfidf_svd_embeddings(
    docs: list[str],
    *,
    max_features: int = 30000,
    n_components: int = 128,
    random_state: int = 42,
) -> tuple[np.ndarray, Any]:
    """
    Lightweight embedding: TF-IDF → TruncatedSVD → L2 normalize rows.
    Returns (doc_embeddings, encode_fn) where encode_fn(texts) -> dense matrix.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    min_df = 1 if len(docs) < 80 else 2
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=0.92,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs)
    n_comp = min(n_components, max(2, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    Z = normalize(svd.fit_transform(X)).astype(np.float64)

    def encode(texts: list[str]) -> np.ndarray:
        Xt = vec.transform(texts)
        return normalize(svd.transform(Xt)).astype(np.float64)

    return Z, encode


def _cluster_sklearn_dbscan(
    embeddings: np.ndarray,
    *,
    min_topic_size: int,
    eps: float,
) -> np.ndarray:
    """Cosine DBSCAN on L2-normalized rows; noise = -1 (same convention as BERTopic outliers)."""
    from sklearn.cluster import DBSCAN

    if min_topic_size < 2:
        min_topic_size = 2
    labels = DBSCAN(
        eps=eps,
        min_samples=min_topic_size,
        metric="cosine",
        n_jobs=-1,
    ).fit_predict(embeddings)
    return np.asarray(labels, dtype=np.int64)


def _fit_bertopic(
    docs: list[str],
    doc_embeddings: np.ndarray,
    embedding_model: Any,
    *,
    min_topic_size: int,
    verbose: bool,
) -> tuple[np.ndarray, Any]:
    from bertopic import BERTopic

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        verbose=verbose,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=doc_embeddings)
    return np.asarray(topics, dtype=np.int64), topic_model


def cluster_subclaims_topic(
    ids: list[str],
    docs: list[str],
    doc_embeddings: np.ndarray,
    embedding_model: Any | None,
    *,
    min_topic_size: int = 2,
    verbose: bool = True,
    backend: str = "auto",
    dbscan_eps: float = 0.32,
) -> tuple[np.ndarray, Any | None, str]:
    """
    Cluster precomputed subclaim embeddings.

    - **bertopic**: needs ``embedding_model`` (SentenceTransformer).
    - **sklearn**: DBSCAN (cosine); works with TF-IDF–SVD or any dense rows.
    - **auto**: try BERTopic when ``embedding_model`` is set; else sklearn; on failure, sklearn.

    Returns ``(topic_ids_per_doc, topic_model_or_none, cluster_backend)``.
    """
    if len(ids) != len(docs):
        raise ValueError("ids and docs must have the same length")
    if len(docs) < 2:
        raise ValueError("need at least 2 documents")

    be = (backend or "auto").strip().lower()

    def _sklearn() -> tuple[np.ndarray, None, str]:
        topics = _cluster_sklearn_dbscan(
            doc_embeddings,
            min_topic_size=min_topic_size,
            eps=dbscan_eps,
        )
        return topics, None, "sklearn_dbscan"

    if be == "sklearn":
        t, m, name = _sklearn()
        return t, m, name

    if be == "bertopic":
        if embedding_model is None:
            print(
                "BERTopic requires sentence-transformers; using sklearn DBSCAN instead.",
                file=sys.stderr,
            )
            return _sklearn()
        try:
            topics, tm = _fit_bertopic(
                docs,
                doc_embeddings,
                embedding_model,
                min_topic_size=min_topic_size,
                verbose=verbose,
            )
            return topics, tm, "bertopic"
        except Exception as e:
            print(
                f"BERTopic failed ({e!r}); using sklearn DBSCAN "
                f"(eps={dbscan_eps}, min_samples={min_topic_size}).",
                file=sys.stderr,
            )
            return _sklearn()

    # auto
    if embedding_model is None:
        return _sklearn()
    try:
        topics, tm = _fit_bertopic(
            docs,
            doc_embeddings,
            embedding_model,
            min_topic_size=min_topic_size,
            verbose=verbose,
        )
        return topics, tm, "bertopic"
    except Exception as e:
        print(
            f"BERTopic unavailable ({e!r}); using sklearn DBSCAN fallback.",
            file=sys.stderr,
        )
        return _sklearn()


def build_bertopic_subclaim_clusters(
    ids: list[str],
    docs: list[str],
    *,
    embedding_model: Any,
    min_topic_size: int = 2,
    verbose: bool = True,
    backend: str = "auto",
    dbscan_eps: float = 0.32,
) -> tuple[np.ndarray, Any | None, np.ndarray, str]:
    """
    Encode with SentenceTransformer, then :func:`cluster_subclaims_topic`.
    Returns ``(topics, topic_model, doc_embeddings, backend_name)``.
    """
    doc_embeddings = _encode_subclaims(docs, embedding_model, verbose=verbose)
    topics, tm, name = cluster_subclaims_topic(
        ids,
        docs,
        doc_embeddings,
        embedding_model,
        min_topic_size=min_topic_size,
        verbose=verbose,
        backend=backend,
        dbscan_eps=dbscan_eps,
    )
    return topics, tm, doc_embeddings, name


def subclaim_rows_from_topics(
    ids: list[str],
    topics: np.ndarray,
    topic_model: Any | None,
) -> dict[str, dict[str, Any]]:
    """Map each subclaim id to topic_id, collapse_flag, collapse_with, and optional topic_label."""
    topics_list = [int(t) for t in topics]
    by_topic: dict[int, list[str]] = {}
    for sid, t in zip(ids, topics_list):
        by_topic.setdefault(t, []).append(sid)

    subclaims_out: dict[str, dict[str, Any]] = {}
    for sid, t in zip(ids, topics_list):
        peers = [x for x in by_topic.get(t, []) if x != sid]
        if t < 0 or len(peers) == 0:
            subclaims_out[sid] = {
                "topic_id": t,
                "collapse_flag": False,
                "collapse_with": [],
            }
        else:
            subclaims_out[sid] = {
                "topic_id": t,
                "collapse_flag": True,
                "collapse_with": sorted(peers),
            }

    if topic_model is not None:
        try:
            topic_info = topic_model.get_topic_info()
            labels_by_id: dict[int, str] = {}
            for _, row in topic_info.iterrows():
                tid = int(row["Topic"])
                if tid < 0:
                    continue
                labels_by_id[tid] = str(row.get("Name", "") or "")
            for sid in subclaims_out:
                tid = int(subclaims_out[sid]["topic_id"])
                if tid >= 0 and tid in labels_by_id:
                    subclaims_out[sid]["topic_label"] = labels_by_id[tid]
        except Exception:
            pass

    return subclaims_out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build BERTopic / DBSCAN collapse artifact for subclaims.",
    )
    parser.add_argument("--claims-json", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--claim-superclaim-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--superclaims-json", type=Path, default=DEFAULT_SUPERCLAIMS)
    parser.add_argument("--codebook-json", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-topic-size", type=int, default=2)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--cluster-backend",
        choices=("auto", "bertopic", "sklearn"),
        default="auto",
        help="Topic clustering: BERTopic, sklearn DBSCAN, or auto (try BERTopic then fall back).",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.32,
        help="Cosine DBSCAN eps when using sklearn (smaller = tighter clusters).",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("auto", "sentence_transformers", "tfidf"),
        default="auto",
        help="Embeddings: MiniLM via sentence-transformers, TF-IDF+SVD (no PyTorch), or auto.",
    )
    args = parser.parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("Missing scikit-learn. Install with: pip install scikit-learn numpy", file=sys.stderr)
        raise SystemExit(1)

    claims_path = args.claims_json
    map_path = args.claim_superclaim_map
    super_path = args.superclaims_json
    codebook_path = args.codebook_json

    for label, p in (
        ("Claims JSON", claims_path),
        ("claim_superclaim_map.json", map_path),
        ("greenwashing_superclaims.json", super_path),
        ("greenwashing_codebook.json", codebook_path),
    ):
        if not p.is_file():
            print(f"{label} not found: {p}", file=sys.stderr)
            return 1

    data_bundle_paths = (claims_path, map_path, super_path, codebook_path)
    data_hashes = [_file_sha256(p) for p in data_bundle_paths]
    bundle_fingerprint = "|".join(data_hashes)
    claims_bundle_version = hashlib.sha256(bundle_fingerprint.encode("utf-8")).hexdigest()[:16]

    claims_sha = _file_sha256(claims_path)

    codebook = _load_id_text_json(codebook_path, "subclaim")
    superclaims = _load_id_text_json(super_path, "superclaim")
    with open(map_path, encoding="utf-8") as f:
        map_raw = json.load(f)
    sub_to_super = _parse_claim_superclaim_map(map_raw)

    with open(claims_path, encoding="utf-8") as f:
        data = json.load(f)

    claims = data.get("claims") or {}
    claims_version_from_file = data.get("claims_version")
    if claims_version_from_file is not None:
        claims_version_from_file = str(claims_version_from_file)

    ids: list[str] = []
    docs: list[str] = []
    collected: set[str] = set()

    def add_sid_doc(sid: str, text: str) -> None:
        t = text.strip()
        if not sid or not t or sid not in sub_to_super:
            return
        if sid in collected:
            return
        collected.add(sid)
        ids.append(sid)
        docs.append(t)

    for claim_id, obj in claims.items():
        sid = (
            str(claim_id)
            if str(claim_id).startswith("NC_")
            else f"NC_{str(claim_id).replace('NC_', '').replace('SC_', '')}"
        )
        text = codebook.get(sid, "").strip() or str((obj or {}).get("current_text") or "").strip()
        add_sid_doc(sid, text)

    for sid in sub_to_super:
        if sid in collected:
            continue
        text = codebook.get(sid, "").strip()
        if text:
            add_sid_doc(sid, text)

    if len(docs) < 2:
        print(
            "Need at least 2 mapped subclaims with non-empty text "
            "(codebook and/or claim history current_text).",
            file=sys.stderr,
        )
        return 1

    eb = args.embedding_backend
    embedding_model: Any | None = None
    encode_fn: Any = None
    doc_embeddings: np.ndarray
    embedding_backend_used: str

    if eb == "tfidf":
        print("Embedding: TF-IDF + TruncatedSVD (no sentence-transformers).")
        doc_embeddings, encode_fn = fit_tfidf_svd_embeddings(docs)
        embedding_backend_used = "tfidf_svd"
        if args.cluster_backend == "bertopic":
            print("BERTopic needs sentence-transformers embeddings; forcing --cluster-backend sklearn.", file=sys.stderr)
            args.cluster_backend = "sklearn"
    elif eb == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print(
                "sentence-transformers not installed. Use --embedding-backend tfidf "
                "or: pip install sentence-transformers",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"Embedding: sentence-transformers ({args.embedding_model})")
        embedding_model = SentenceTransformer(args.embedding_model)
        doc_embeddings = _encode_subclaims(docs, embedding_model, verbose=True)
        encode_fn = lambda texts: embedding_model.encode(  # type: ignore[misc]
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedding_backend_used = "sentence_transformers"
    else:
        # auto: prefer sentence-transformers; fall back to TF-IDF
        try:
            from sentence_transformers import SentenceTransformer

            print(f"Embedding: sentence-transformers ({args.embedding_model})")
            embedding_model = SentenceTransformer(args.embedding_model)
            doc_embeddings = _encode_subclaims(docs, embedding_model, verbose=True)
            encode_fn = lambda texts: embedding_model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embedding_backend_used = "sentence_transformers"
        except Exception as e:
            print(f"sentence-transformers unavailable ({e!r}); using TF-IDF + SVD.", file=sys.stderr)
            doc_embeddings, encode_fn = fit_tfidf_svd_embeddings(docs)
            embedding_model = None
            embedding_backend_used = "tfidf_svd"
            if args.cluster_backend == "bertopic":
                args.cluster_backend = "sklearn"

    print(f"claims_bundle_version: {claims_bundle_version}")
    print(
        f"Clustering {len(docs)} subclaims "
        f"(cluster={args.cluster_backend}, embedding={embedding_backend_used}, min_topic_size={args.min_topic_size})…",
    )

    topics, topic_model, cluster_backend = cluster_subclaims_topic(
        ids,
        docs,
        doc_embeddings,
        embedding_model,
        min_topic_size=args.min_topic_size,
        verbose=True,
        backend=args.cluster_backend,
        dbscan_eps=args.dbscan_eps,
    )
    print(f"Cluster backend used: {cluster_backend}")
    subclaims_out = subclaim_rows_from_topics(ids, topics, topic_model)

    sid_to_doc = dict(zip(ids, docs))
    for sid in ids:
        scid = sub_to_super.get(sid)
        if not scid or sid not in subclaims_out:
            continue
        sub_text = sid_to_doc.get(sid, "").strip()
        super_text = superclaims.get(scid, "").strip()
        if not sub_text or not super_text:
            continue
        pair_emb = encode_fn([sub_text, super_text])
        sim = float(np.dot(pair_emb[0], pair_emb[1]))
        sim = max(0.0, min(1.0, sim))
        subclaims_out[sid]["hierarchy_confidence"] = round(sim, 4)
        subclaims_out[sid]["superclaim_id"] = scid

    out_obj: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claims_bundle_version": claims_bundle_version,
        "claims_source_sha256": claims_sha,
        "data_files_sha256": data_hashes,
        "claims_version": claims_version_from_file,
        "cluster_backend": cluster_backend,
        "embedding_backend": embedding_backend_used,
        "embedding_model": args.embedding_model if embedding_backend_used == "sentence_transformers" else None,
        "min_topic_size": args.min_topic_size,
        "dbscan_eps": args.dbscan_eps if cluster_backend == "sklearn_dbscan" else None,
        "claims_source": os.path.basename(str(claims_path)),
        "subclaims": subclaims_out,
    }

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    flagged = sum(1 for v in subclaims_out.values() if v.get("collapse_flag"))
    with_hier = sum(1 for v in subclaims_out.values() if "hierarchy_confidence" in v)
    print(f"Wrote {out_path}")
    print(f"  collapse_flag=true: {flagged} subclaims; hierarchy_confidence: {with_hier} subclaims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
