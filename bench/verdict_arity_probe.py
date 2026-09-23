"""Option-count (arity) probe of openJev-verdict-2.0 on the 24-way command question.

The zero-shot runs showed the 151M GLiClass encoder putting every request on one label once the
command question carries its full catalog. This probe separates two candidate causes:

  * option COUNT: GLiClass scores every candidate slot against one packed prompt; its authors
    evaluate on 5-candidate tasks and the calibrator carries per-k temperatures up to k = 25;
  * WINDOW TRUNCATION: the prompt is "<<LABEL>>desc ... <<SEP>>Question ... Context: state", cut
    to max_length = 512 from the right, so long option descriptions push the state itself out.

For each canonical request the gold command is kept and k-1 distractors are drawn deterministically
from the rest of the catalog (catalog order preserved), for k in --ks, in three description modes
(full catalog description / first clause / bare option name). One question per call, ONNX CPU, the
shipped engine defaults. Also replays the study's original four-request probe verbatim.

    python -m bench.verdict_arity_probe --models ~/projects/jevlet-work/models \
        --repo ~/projects/jevlet-work/verdict-open-jev --out results/verdict-arity-probe.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics
import sys
import time

from jevlet.backends.encoders import VerdictBackend, _desc
from jevlet.schema import Question, normalize_questions
from bench import netops


def entropy_bits(counter: dict) -> float:
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c)


def subset_for(row_i: int, k: int, gold: str, catalog: list[str]) -> list[str]:
    """gold + (k-1) distractors, deterministic per (row, k), in catalog order."""
    if k >= len(catalog):
        return list(catalog)
    others = [o for o in catalog if o != gold]
    rng = random.Random(1000 * row_i + k)
    picked = set(rng.sample(others, k - 1)) | {gold}
    return [o for o in catalog if o in picked]


def prompt_tokens(be: VerdictBackend, q: Question, state: str, mode: str) -> int:
    from core.formatting import build_model_input
    labels = [f"It is {_desc(q, o, mode)}" for o in q.options] + ["insufficient evidence"]
    return len(be.engine.tokenizer(build_model_input(q.instructions, state, labels), truncation=False)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--splits", default="canonical")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ks", default="6,12,16,20,24")
    ap.add_argument("--modes", default="full,short,bare")
    ap.add_argument("--out", default="results/verdict-arity-probe.json")
    a = ap.parse_args()

    ks = [int(x) for x in a.ks.split(",")]
    modes = a.modes.split(",")
    rows = netops.load_rows(tuple(a.splits.split(",")))
    rows = [r for r in rows if r.get("expected_command")]
    if a.limit:
        rows = rows[: a.limit]
    base_q = {q.qid: q for q in normalize_questions(netops.router_questions())}["command"]
    catalog = list(base_q.options)

    models = os.path.expanduser(a.models)
    repo = os.path.expanduser(a.repo)
    be = VerdictBackend(os.path.join(models, "heman10x_rlcd-modernbert-151m"), repo, "cpu", use_onnx=True, desc_mode="short")
    cal = getattr(be.engine, "calibrator", None)
    per_k = dict(getattr(cal, "per_k", {}) or {}) if cal is not None else {}

    out = {"meta": {"backend": be.name, "engine": "ONNX CPU, shipped DecisionEngine defaults (intra_op 4 threads)",
                    "max_length": be.engine.max_length, "host": os.uname().machine, "cpu": os.cpu_count(),
                    "n_rows": len(rows), "splits": a.splits, "ks": ks, "modes": modes, "catalog": catalog,
                    "calibrator_per_k": per_k, "question": base_q.instructions},
           "conditions": [], "per_row": [], "replication": []}

    t_all = time.time()
    for mode in modes:
        for k in ks:
            preds = collections.Counter()
            correct = 0
            maxps, abst, lat, ntoks = [], [], [], []
            for i, r in enumerate(rows):
                opts = subset_for(i, k, r["expected_command"], catalog)
                q = Question("command", "choice", base_q.instructions, opts,
                             {o: base_q.descriptions.get(o) for o in opts if base_q.descriptions.get(o)})
                be.desc_mode = mode
                t0 = time.perf_counter()
                ans = be._decide(r["input"], [q])[0]
                lat.append(time.perf_counter() - t0)
                j = max(range(len(opts)), key=lambda x: ans.probs[x])
                pred = opts[j]
                preds[pred] += 1
                correct += pred == r["expected_command"]
                maxps.append(ans.probs[j])
                abst.append(ans.meta.get("p_abstain", 0.0))
                nt = prompt_tokens(be, q, r["input"], mode)
                ntoks.append(nt)
                out["per_row"].append({"mode": mode, "k": k, "i": i, "gold": r["expected_command"], "pred": pred,
                                       "max_p": round(ans.probs[j], 4), "p_abstain": round(abst[-1], 4), "prompt_tokens": nt})
            n = len(rows)
            cond = {"mode": mode, "k": k, "n": n, "accuracy": correct / n, "chance": 1 / k,
                    "modal_label": preds.most_common(1)[0][0], "modal_share": preds.most_common(1)[0][1] / n,
                    "distinct_labels": len(preds), "pred_entropy_bits": entropy_bits(preds), "max_entropy_bits": math.log2(k),
                    "mean_max_p": statistics.mean(maxps), "mean_p_abstain": statistics.mean(abst),
                    "prompt_tokens_median": statistics.median(ntoks), "prompt_tokens_max": max(ntoks),
                    "frac_truncated": sum(t > be.engine.max_length for t in ntoks) / n,
                    "latency_ms_median": statistics.median(lat) * 1000,
                    "top_preds": preds.most_common(3)}
            out["conditions"].append(cond)
            print(f"{mode:5s} k={k:2d} acc={cond['accuracy']:.3f} chance={1/k:.3f} modal={cond['modal_share']:.2f} "
                  f"({cond['modal_label']}) H={cond['pred_entropy_bits']:.2f}/{math.log2(k):.2f} bits "
                  f"tok med/max {cond['prompt_tokens_median']:.0f}/{cond['prompt_tokens_max']} trunc={cond['frac_truncated']:.2f} "
                  f"{cond['latency_ms_median']:.0f} ms", flush=True)

    # the study's original probe, verbatim: four requests, first-n catalog options, short descriptions
    from rlcd import Choice, Option
    states = ["@sw1 disable port 3", "@sw1 show port 5 status", "@sw1 set port 2 speed to 1000", "@sw1 turn off poe on port 4"]
    for n in (6, 12, 16, 20, 22, 23, 24):
        opts = catalog[:n]
        if "disable_port" not in opts:
            opts = opts[:-4] + ["enable_port", "disable_port", "configure_poe_mode", "configure_port_speed"]
        ch = Choice(id="c", question=base_q.instructions,
                    options=tuple(Option(id=o, description=_desc(base_q, o, "short")) for o in opts))
        picks = [be.engine.evaluate(s, [ch]).results[0].selected_id for s in states]
        out["replication"].append({"n_options": n, "options": opts, "states": states, "picks": picks,
                                   "distinct_picks": len(set(picks))})
        print("replication", n, picks, flush=True)

    out["meta"]["elapsed_s"] = time.time() - t_all
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, f"{out['meta']['elapsed_s']:.0f} s")


if __name__ == "__main__":
    main()
