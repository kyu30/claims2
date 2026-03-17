import argparse
import json
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from ollama import chat

MODEL = 'gpt-oss:20b'  # change to your local model name

SUPERCLAIMS_PATH = "greenwashing_superclaims.json"
SUBCLAIMS_PATH = "greenwashing_codebook.json"
MAPPINGS_PATH = "claim_superclaim_map.json"

TOP_K_CANDIDATES = 8

SYSTEM = (
    "You are a careful claim taxonomy validator.\n"
    "Return ONLY valid JSON. No markdown, no extra text.\n"
)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_dict_claims(obj: Any, name: str) -> Dict[str, str]:
    if isinstance(obj, dict):
        return {str(k): str(v).strip() for k, v in obj.items()}
    raise ValueError(f"{name} must be a JSON object/dict of {{id: text}}")

def normalize_mappings(obj: Any) -> List[Tuple[str, str]]:
    """
    Supports:
      1) dict: {NC_ID: SC_ID}
      2) list of dicts with keys like subclaim_id/superclaim_id
      3) list of pairs: [["NC_1","SC_1"], ...]
    """
    if isinstance(obj, dict):
        return [(str(nc), str(sc)) for nc, sc in obj.items()]

    if isinstance(obj, list):
        out: List[Tuple[str, str]] = []
        for item in obj:
            if isinstance(item, dict):
                nc = item.get("subclaim_id") or item.get("nc_id") or item.get("subclaim") or item.get("NC")
                sc = item.get("superclaim_id") or item.get("sc_id") or item.get("superclaim") or item.get("SC")
                if nc is None or sc is None:
                    raise ValueError(f"Mapping object missing keys: {item}")
                out.append((str(nc), str(sc)))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((str(item[0]), str(item[1])))
            else:
                raise ValueError(f"Unsupported mapping entry: {item}")
        return out

    raise ValueError("mappings.json must be a dict or list")


def load_combined_input(path: str) -> Tuple[Dict[str, str], Dict[str, str], List[Tuple[str, str]]]:
    """Load a combined JSON file of the form:
    {
      "SUB_ID": {
        "superclaim_id": "SC_...",
        "superclaim_text": "...",
        "subclaim_text": "..."
      },
      ...
    }

    Returns (superclaims, subclaims, mappings)
    """
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at top level in {path}")

    superclaims: Dict[str, str] = {}
    subclaims: Dict[str, str] = {}
    mappings: List[Tuple[str, str]] = []

    for sub_id, record in data.items():
        if not isinstance(record, dict):
            raise ValueError(f"Expected object for subclaim {sub_id}, got {type(record)}")

        sc_id = (
            record.get("superclaim_id")
            or record.get("superclaimId")
            or record.get("sc_id")
            or record.get("SC")
        )
        sc_text = (
            record.get("superclaim_text")
            or record.get("superclaimText")
            or record.get("superclaim")
        )
        sub_text = (
            record.get("subclaim_text")
            or record.get("subclaimText")
            or record.get("subclaim")
        )

        if sc_id is None or sc_text is None or sub_text is None:
            raise ValueError(
                f"Missing fields for subclaim {sub_id}. Requires superclaim_id, superclaim_text, and subclaim_text."
            )

        sc_id = str(sc_id)
        sc_text = str(sc_text).strip()
        sub_text = str(sub_text).strip()

        if sc_id in superclaims and superclaims[sc_id] != sc_text:
            print(f"Warning: inconsistent superclaim text for {sc_id}; using first encountered")
        else:
            superclaims[sc_id] = sc_text

        subclaims[str(sub_id)] = sub_text
        mappings.append((str(sub_id), sc_id))

    return superclaims, subclaims, mappings

def simple_topk_candidates(subclaim_text: str, superclaims: Dict[str, str], k: int) -> List[str]:
    # lightweight token overlap retrieval (no extra deps)
    def tokens(s: str) -> set:
        return set(re.findall(r"[a-zA-Z]{3,}", s.lower()))
    q = tokens(subclaim_text)
    scored = []
    for sid, txt in superclaims.items():
        t = tokens(txt)
        score = len(q & t) / (1 + len(q))
        scored.append((score, sid))
    scored.sort(reverse=True)
    return [sid for _, sid in scored[:k]]

def make_prompt(
    sub_id: str,
    sub_text: str,
    mapped_super_id: str,
    mapped_super_text: str,
    candidate_super_ids: List[str],
    superclaims: Dict[str, str],
) -> str:
    candidates_block = [{"id": sid, "text": superclaims[sid]} for sid in candidate_super_ids]
    return f"""
Task: Validate whether SUBCLAIM belongs under the given SUPERCLAIM in a claim hierarchy.

VALID if the subclaim clearly supports, instantiates, refines, or is a component of the superclaim.
INVALID if out of scope, only loosely related, a sibling claim, or mismatched in "type"
(e.g., descriptive evidence mapped to an unrelated normative conclusion).

SUBCLAIM:
- id: {sub_id}
- text: {sub_text}

MAPPED SUPERCLAIM:
- id: {mapped_super_id}
- text: {mapped_super_text}

CANDIDATE SUPERCLAIMS (choose from these if suggesting a better parent):
{json.dumps(candidates_block, ensure_ascii=False, indent=2)}

Return JSON with exactly these keys:
{{
  "verdict": "valid" | "invalid" | "uncertain",
  "confidence": 0.0-1.0,
  "reason": "short explanation",
  "suggested_superclaim_id": string|null,
  "suggested_superclaim_reason": string|null
}}

Rules:
- If verdict is "valid": suggested_superclaim_id MUST be null.
- If verdict is "invalid" and a better match exists among candidates: pick that id and briefly explain why it's better
- If none fit: suggested_superclaim_id null and explain.
""".strip()

def call_ollama_json(prompt: str) -> Dict[str, Any]:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.1, 'num_predict': 512},
    )
    text = resp["message"]["content"].strip()

    # defensively extract JSON object if needed
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Model did not return JSON. Got: {text[:500]}")
    return json.loads(m.group(0))


def mock_call_ollama_json(prompt: str) -> Dict[str, Any]:
    """Mock response for testing without Ollama."""
    verdicts = ["valid", "invalid", "uncertain"]
    verdict = random.choice(verdicts)
    confidence = random.uniform(0.5, 1.0) if verdict != "uncertain" else random.uniform(0.0, 0.5)
    reason = f"Mock reason for {verdict} verdict."
    suggested_id = random.choice([None, "SC_1", "SC_2"]) if verdict == "invalid" else None
    suggested_reason = f"Mock suggestion reason." if suggested_id else None
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
        "suggested_superclaim_id": suggested_id,
        "suggested_superclaim_reason": suggested_reason,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Validate subclaim->superclaim mappings (supports combined JSON input)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "(Positional) Path to a combined JSON file with structure {subclaim_id: {superclaim_id, superclaim_text, subclaim_text}}. "
            "If omitted, the script will load separate superclaims/subclaims/mappings files."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input",
        help="Alias for the positional input file.",
    )
    parser.add_argument(
        "--superclaims",
        default=SUPERCLAIMS_PATH,
        help="Legacy: path to superclaims JSON.",
    )
    parser.add_argument(
        "--subclaims",
        default=SUBCLAIMS_PATH,
        help="Legacy: path to subclaims JSON.",
    )
    parser.add_argument(
        "--mappings",
        default=MAPPINGS_PATH,
        help="Legacy: path to mappings JSON.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock responses instead of calling Ollama model (for testing).",
    )
    args = parser.parse_args()

    if args.input:
        superclaims, subclaims, mappings = load_combined_input(args.input)
    else:
        superclaims = normalize_dict_claims(load_json(args.superclaims), args.superclaims)
        subclaims = normalize_dict_claims(load_json(args.subclaims), args.subclaims)
        mappings = normalize_mappings(load_json(args.mappings))

    validated_rows = []
    flagged_rows = []

    # superclaim -> list[subclaim_id] (ALL mappings)
    super_to_sub_all: Dict[str, List[str]] = defaultdict(list)
    for nc_id, sc_id in mappings:
        super_to_sub_all[sc_id].append(nc_id)

    for nc_id, sc_id in mappings:
        print(f"Checking subclaim {nc_id}")
        sub_text = subclaims.get(nc_id, "").strip()
        sup_text = superclaims.get(sc_id, "").strip()

        if not sub_text or not sup_text:
            flagged_rows.append({
                "subclaim_id": nc_id,
                "subclaim_id_missing": not bool(sub_text),
                "superclaim_id": sc_id,
                "superclaim_id_missing": not bool(sup_text),
                "verdict": "invalid",
                "confidence": 1.0,
                "reason": "Missing text for subclaim or superclaim id.",
                "suggested_superclaim_id": None,
                "suggested_superclaim_reason": None,
            })
            continue

        candidate_ids = simple_topk_candidates(sub_text, superclaims, TOP_K_CANDIDATES)
        if sc_id not in candidate_ids:
            candidate_ids = [sc_id] + candidate_ids[:-1]

        prompt = make_prompt(nc_id, sub_text, sc_id, sup_text, candidate_ids, superclaims)
        try:
            if args.mock:
                result = mock_call_ollama_json(prompt)
            else:
                result = call_ollama_json(prompt)
        except Exception as e:
            print(f"Warning: Failed to get response for {nc_id}: {e}")
            result = {
                "verdict": "uncertain",
                "confidence": 0.0,
                "reason": f"Model error: {str(e)}",
                "suggested_superclaim_id": None,
                "suggested_superclaim_reason": None,
            }

        row = {
            "subclaim_id": nc_id,
            "subclaim_text": sub_text,
            "superclaim_id": sc_id,
            "superclaim_text": sup_text,
            **result,
        }

        if result.get("verdict") == "valid":
            validated_rows.append(row)
        else:
            flagged_rows.append(row)

    # outputs
    with open("validated_mappings.jsonl", "w", encoding="utf-8") as f:
        for r in validated_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open("flagged_mappings.jsonl", "w", encoding="utf-8") as f:
        for r in flagged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open("superclaim_to_subclaims.json", "w", encoding="utf-8") as f:
        json.dump(super_to_sub_all, f, ensure_ascii=False, indent=2)

    super_to_sub_flagged: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in flagged_rows:
        # keep text for convenience
        super_to_sub_flagged[r["superclaim_id"]].append({"id": r["subclaim_id"], "text": r.get("subclaim_text", "")})

    with open("superclaim_to_subclaims_flagged_only.json", "w", encoding="utf-8") as f:
        json.dump(super_to_sub_flagged, f, ensure_ascii=False, indent=2)

    print(f"Done. Valid: {len(validated_rows)} | Flagged/uncertain: {len(flagged_rows)}")

if __name__ == "__main__":
    main()