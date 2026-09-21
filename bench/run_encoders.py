"""Experiment 2 — open encoder decision models, zero-shot, on the same batteries as the readout.

    python -m bench.run_encoders --backend verdict --models ~/projects/jevlet-work/models \
        --repo ~/projects/jevlet-work/verdict-open-jev --tag verdict-onnx-cpu [--device cpu] [--gate --run ...]

Output files share the schema of bench.run_readout so bench.analyze reads both.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from jevlet.backends.encoders import make_backend
from jevlet.schema import normalize_questions
from bench import netops


def run_router(backend, rows, out_path, questions=None):
    qs = normalize_questions(questions or netops.router_questions())
    results = {q.qid: [] for q in qs}
    lat = []
    t0 = time.time()
    for i, r in enumerate(rows):
        t1 = time.perf_counter()
        ans = backend._decide(r["input"], qs)      # all questions in ONE forward pass
        lat.append(time.perf_counter() - t1)
        for a in ans:
            results[a.qid].append({"i": i, "probs": a.probs, "logits": a.logits, "latency_s": a.meta.get("latency_s"),
                                   "label_mass": 1.0, "extra": {k: v for k, v in a.meta.items() if k != "latency_s"}})
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    meta = {"backend": backend.name, "questions": {q.qid: {"kind": q.kind, "options": q.options} for q in qs},
            "rows": [{k: r.get(k) for k in ("input", "expected_tool", "expected_command", "expected_params", "style",
                                             "base_input", "split", "is_write", "touches_poe_power", "ambiguity",
                                             "alt_ok", "dangerous_pair", "level")} for r in rows],
            "elapsed_s": time.time() - t0, "call_latency_s": lat}
    json.dump({"meta": meta, "results": results}, open(out_path, "w"))
    print("wrote", out_path, f"median call {sorted(lat)[len(lat)//2]*1000:.0f} ms")


def run_gate(backend, pairs, out_path):
    qs = normalize_questions(netops.gate_questions())
    res = []
    t0 = time.time()
    for i, p in enumerate(pairs):
        state = netops.gate_state(p["input"], p["command"], p["params"])
        a = backend._decide(state, qs)[0]
        res.append({**p, "p_yes": a.probs[1], "logits": a.logits, "latency_s": a.meta.get("latency_s")})
        if i % 200 == 0:
            print(f"  [gate] {i}/{len(pairs)}  {time.time()-t0:.0f}s", flush=True)
    json.dump({"meta": {"backend": backend.name, "elapsed_s": time.time() - t0}, "results": res}, open(out_path, "w"))
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", default="canonical,paraphrase,authored")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    Path(a.outdir).mkdir(exist_ok=True)
    be = make_backend(a.backend, a.models, a.repo, a.device)
    rows = netops.load_rows(tuple(a.splits.split(",")))
    if a.limit:
        rows = rows[: a.limit]
    if a.gate:
        pairs = netops.gate_pairs(a.run, rows=[r for r in rows if r["split"] in ("canonical", "paraphrase")])
        if a.limit:
            pairs = pairs[: a.limit]
        run_gate(be, pairs, f"{a.outdir}/gate-{a.tag}.json")
    else:
        run_router(be, rows, f"{a.outdir}/router-{a.tag}.json")
