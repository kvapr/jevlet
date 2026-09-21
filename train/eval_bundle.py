"""Evaluate a fine-tuned bundle (typed-decisions format) on the held-out test GROUPS with the
same router / gate batteries and result schema as the other experiments, so bench.analyze
compares it with the readout and the zero-shot encoders.

    python -m train.eval_bundle --bundle runs/modernbert-base-desc/bundle --pkg $WORK/typed-decisions/src \
        --split-manifest data/td-desc/split_manifest.json --tag ft-modernbert-base [--device cuda] [--no-desc]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlet.backends.encoders import OpenJevBackend
from bench import netops
from bench.run_encoders import run_router, run_gate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-desc", action="store_true")
    ap.add_argument("--run", default="data/netops/generative_runs/g3-2507-npu-para195.json")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    Path(a.outdir).mkdir(exist_ok=True)
    test_groups = set(json.load(open(a.split_manifest))["test_groups"])
    rows = [r for r in netops.load_rows() if r["base_input"] in test_groups]
    be = OpenJevBackend(a.bundle, a.device, pkg_dir=a.pkg)
    be.name = a.tag
    qs = netops.router_questions()
    if a.no_desc:  # bare option names, as trained
        for qid in ("tool", "command"):
            qs[qid]["criteria"] = {k: None for k in qs[qid]["criteria"]}
    run_router(be, rows, f"{a.outdir}/router-{a.tag}.json", questions=qs)
    pairs = netops.gate_pairs(a.run, rows=[r for r in rows if r["split"] in ("canonical", "paraphrase")])
    run_gate(be, pairs, f"{a.outdir}/gate-{a.tag}.json")
    print(f"evaluated {len(rows)} held-out rows / {len(pairs)} gate pairs")
