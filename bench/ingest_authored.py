"""Turn the data-authoring workflow's JSON result into the authored YAML files + a review log.

    python -m bench.ingest_authored <workflow-output.json>

Writes data/authored/<category>.yaml (evaluation rows, verified), data/authored/train_paraphrases.yaml
(training-only, label-preserving paraphrases, drift-checked) and docs/data-review-log.md.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "data" / "authored"
DOCS = ROOT / "docs"

PORT_FIX = re.compile(r"^(\d{1,2})$")


def _clean_row(r: dict) -> dict:
    from bench.netops import _sanitize_text
    r = dict(r)
    r["input"] = _sanitize_text(r["input"])
    params = r.get("expected_params") or {}
    if isinstance(params, dict) and isinstance(params.get("port"), (str, int)) and PORT_FIX.match(str(params["port"])):
        params["port"] = f"1/{params['port']}"
    r["expected_params"] = params
    r["expected_command"] = r.get("expected_command") or ""
    r["alt_ok"] = [a for a in (r.get("alt_ok") or []) if a]
    return r


def main(path):
    d = json.load(open(path))
    res = d["result"] if "result" in d else d
    AUTH.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    log = ["# Data review log", "", "Rows were written by one agent and verified by an independent agent; every change is listed.",
           "Human review pending (Krish). Rows with ambiguity >= 2 are deliberately included: they are the cases a calibrated model should be unsure about.", ""]
    counts = {}
    for cat in res["authored"]:
        rows = [_clean_row(r) for r in cat["rows"]]
        name = cat["category"]
        yaml.safe_dump(rows, open(AUTH / f"{name}.yaml", "w"), sort_keys=False, allow_unicode=True)
        counts[name] = len(rows)
        log.append(f"## {name} ({len(rows)} rows)")
        log += [f"- {l}" for l in cat.get("review_log", [])]
        if cat.get("dropped"):
            log.append("- dropped: " + "; ".join(cat["dropped"]))
        log.append("")
    from bench.netops import _sanitize_text
    paras = [{"base_input": _sanitize_text(p["base_input"]), "input": _sanitize_text(p["input"]),
              "style": p["style"]} for p in res["paraphrases"]]
    log = [_sanitize_text(l) for l in log]
    yaml.safe_dump(paras, open(AUTH / "train_paraphrases.yaml", "w"), sort_keys=False, allow_unicode=True)
    counts["train_paraphrases"] = len(paras)
    log.append(f"## train_paraphrases ({len(paras)} rows, training only)")
    log += [f"- {l}" for l in res.get("para_logs", [])[:200]]
    log = [_sanitize_text(l) for l in log]          # every line, including the drift-check notes
    open(DOCS / "data-review-log.md", "w").write("\n".join(log) + "\n")
    print(counts)


if __name__ == "__main__":
    main(sys.argv[1])
