"""Experiment 1 — next-token readout of an instruction-tuned LM as a System One backend.

Runs the router battery (tool / command / is_write / touches_poe_power) and the confirm-gate
battery (matches_request) against a llama-server, question-major so the cached prefix is reused,
and stores raw probabilities + log-probs per row for later calibration and analysis.

    python -m bench.run_readout --url http://127.0.0.1:18999 --tag qwen3-4b-q4_0-cpu --rotations 3
    python -m bench.run_readout --gate --run data/netops/generative_runs/g3-2507-npu-para195.json ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from jevlet.backends.llamacpp import LlamaCppReadout
from jevlet.schema import normalize_questions
from bench import netops

DOMAIN = ("You are the dispatcher of a network-switch management assistant. Operators mention the switch as @sw1; "
          "'port N' means interface 1/N.")


def run_router(backend, rows, out_path, questions=None):
    qs = normalize_questions(questions or netops.router_questions())
    results = {q.qid: [] for q in qs}
    t0 = time.time()
    for q in qs:
        n_rot = 1 if q.kind == "noul" else min(backend.rotations, q.k)
        per_row = [[] for _ in rows]
        # rotation-major: the system prompt is identical across rows within a rotation -> prefix cache hits
        for r in range(n_rot):
            for i, row in enumerate(rows):
                per_row[i].append(backend.readout_rotation(row["input"], q, r))
                if i % 100 == 0:
                    print(f"  [{q.qid} rot{r}] {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
        for i, rots in enumerate(per_row):
            probs, logs = backend.combine_rotations(rots)
            results[q.qid].append({"i": i, "probs": probs, "logits": logs,
                                   "per_rotation": [x["probs"] for x in rots],
                                   "label_mass": sum(x["label_mass"] for x in rots) / len(rots),
                                   "latency_s": sum(x["latency_s"] for x in rots),
                                   "prompt_tokens": rots[0].get("prompt_tokens"),
                                   "first_token": rots[0].get("first_token")})
    meta = {"backend": backend.name, "url": backend.base_url, "rotations": backend.rotations,
            "questions": {q.qid: {"kind": q.kind, "options": q.options} for q in qs},
            "rows": [{k: r.get(k) for k in ("input", "expected_tool", "expected_command", "expected_params", "style",
                                             "base_input", "split", "is_write", "touches_poe_power", "ambiguity",
                                             "alt_ok", "dangerous_pair", "level")} for r in rows],
            "elapsed_s": time.time() - t0}
    json.dump({"meta": meta, "results": results}, open(out_path, "w"))
    print("wrote", out_path)


def run_gate(backend, pairs, out_path):
    qs = normalize_questions(netops.gate_questions())
    res = []
    t0 = time.time()
    for i, p in enumerate(pairs):
        state = netops.gate_state(p["input"], p["command"], p["params"])
        a = backend._decide(state, qs)[0]
        res.append({**p, "p_yes": a.probs[1], "logits": a.logits, "label_mass": a.meta["label_mass"],
                    "latency_s": a.meta["latency_s"]})
        if i % 100 == 0:
            print(f"  [gate] {i}/{len(pairs)}  {time.time()-t0:.0f}s", flush=True)
    json.dump({"meta": {"backend": backend.name, "url": backend.base_url, "elapsed_s": time.time() - t0},
               "results": res}, open(out_path, "w"))
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18999")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rotations", type=int, default=1)
    ap.add_argument("--splits", default="canonical,paraphrase,authored")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--run", default=None, help="generative run json for real gate pairs")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    Path(a.outdir).mkdir(exist_ok=True)
    be = LlamaCppReadout(a.url, rotations=a.rotations, domain_hint=DOMAIN)
    rows = netops.load_rows(tuple(a.splits.split(",")))
    if a.limit:
        rows = rows[: a.limit]
    if a.gate:
        pairs = netops.gate_pairs(a.run, rows=[r for r in rows if r["split"] in ("canonical", "paraphrase")],
                                  synthetic=not a.no_synthetic)
        if a.limit:
            pairs = pairs[: a.limit]
        run_gate(be, pairs, f"{a.outdir}/gate-{a.tag}.json")
    else:
        run_router(be, rows, f"{a.outdir}/router-{a.tag}-rot{a.rotations}.json")
