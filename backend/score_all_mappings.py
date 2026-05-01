import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from llm_confidence import score_subclaim_to_superclaim_confidence


ROOT = Path(__file__).resolve().parent

CODEBOOK_NAME = "greenwashing_codebook.json"
SUPERCLAIMS_NAME = "greenwashing_superclaims.json"
MAP_NAME = "claim_superclaim_map.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_subclaim_id(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if s.startswith("NC_"):
        return s
    if s.startswith("SC_"):
        s = s.replace("SC_", "", 1)
    return f"NC_{s}"


def _normalize_superclaim_id(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if s.startswith("SC_"):
        return s
    if s.startswith("NC_"):
        s = s.replace("NC_", "", 1)
    return f"SC_{s}"


def _iter_mappings(mapping_obj: Any) -> Iterable[Tuple[str, str]]:
    if isinstance(mapping_obj, dict):
        for k, v in mapping_obj.items():
            yield (_normalize_subclaim_id(str(k)), _normalize_superclaim_id(str(v)))
        return
    if isinstance(mapping_obj, list):
        for item in mapping_obj:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                yield (_normalize_subclaim_id(str(item[0])), _normalize_superclaim_id(str(item[1])))
                continue
            if isinstance(item, dict):
                nc = (
                    item.get("subclaim_id")
                    or item.get("subclaimId")
                    or item.get("nc_id")
                    or item.get("NC")
                    or item.get("subclaim")
                )
                sc = (
                    item.get("superclaim_id")
                    or item.get("superclaimId")
                    or item.get("sc_id")
                    or item.get("SC")
                    or item.get("superclaim")
                )
                if nc is None or sc is None:
                    continue
                yield (_normalize_subclaim_id(str(nc)), _normalize_superclaim_id(str(sc)))
                continue
        return
    raise ValueError("claim_superclaim_map.json must be a dict or list")


def _pair_key(subclaim_id: str, superclaim_id: str, subclaim_text: str, superclaim_text: str) -> str:
    h = hashlib.sha256()
    h.update(subclaim_id.encode("utf-8"))
    h.update(b"\0")
    h.update(superclaim_id.encode("utf-8"))
    h.update(b"\0")
    h.update(subclaim_text.encode("utf-8"))
    h.update(b"\0")
    h.update(superclaim_text.encode("utf-8"))
    return h.hexdigest()[:16]


def _load_existing_keys(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    keys: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            k = str(obj.get("pair_key") or "").strip()
            if k:
                keys.add(k)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score confidence for every NC_* -> SC_* mapping using an OpenAI LLM call."
    )
    parser.add_argument("--taxonomy-dir", default=str(ROOT), help="Directory containing the three taxonomy JSON files.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "claim_superclaim_map_confidence.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(os.getenv("OPENAI_SCORE_SLEEP_SECONDS") or 0.0),
        help="Optional delay between LLM calls (helps rate limits).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, skip pairs already present (by pair_key).",
    )
    args = parser.parse_args()

    tax_dir = Path(args.taxonomy_dir).resolve()
    codebook_path = tax_dir / CODEBOOK_NAME
    superclaims_path = tax_dir / SUPERCLAIMS_NAME
    map_path = tax_dir / MAP_NAME
    out_path = Path(args.out).resolve()

    codebook_obj = _read_json(codebook_path)
    superclaims_obj = _read_json(superclaims_path)
    mapping_obj = _read_json(map_path)

    if not isinstance(codebook_obj, dict):
        raise ValueError(f"{codebook_path} must be a JSON object of {{NC_*: text}}")
    if not isinstance(superclaims_obj, dict):
        raise ValueError(f"{superclaims_path} must be a JSON object of {{SC_*: text}}")

    existing = _load_existing_keys(out_path) if args.resume else set()
    total = 0
    skipped = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out:
        for nc_id, sc_id in _iter_mappings(mapping_obj):
            if not nc_id or not sc_id:
                continue

            sub_text = str(codebook_obj.get(nc_id) or "").strip()
            sup_text = str(superclaims_obj.get(sc_id) or "").strip()
            total += 1

            if not sub_text or not sup_text:
                row = {
                    "pair_key": _pair_key(nc_id, sc_id, sub_text, sup_text),
                    "subclaim_id": nc_id,
                    "subclaim_text": sub_text,
                    "superclaim_id": sc_id,
                    "superclaim_text": sup_text,
                    "verdict": "invalid",
                    "confidence": 1.0,
                    "reason": "Missing subclaim_text or superclaim_text for this id.",
                    "model": None,
                }
                if row["pair_key"] in existing:
                    skipped += 1
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                existing.add(row["pair_key"])
                continue

            k = _pair_key(nc_id, sc_id, sub_text, sup_text)
            if k in existing:
                skipped += 1
                continue

            score = score_subclaim_to_superclaim_confidence(
                subclaim_text=sub_text,
                superclaim_text=sup_text,
                subclaim_id=nc_id,
                superclaim_id=sc_id,
            )

            row = {
                "pair_key": k,
                "subclaim_id": nc_id,
                "subclaim_text": sub_text,
                "superclaim_id": sc_id,
                "superclaim_text": sup_text,
                "verdict": score.get("verdict"),
                "confidence": score.get("confidence"),
                "reason": score.get("reason"),
                "model": score.get("model"),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            existing.add(k)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"Done. Total seen: {total}. Skipped (resume): {skipped}. Output: {out_path}")


if __name__ == "__main__":
    main()

