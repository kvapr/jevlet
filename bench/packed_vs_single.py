"""Packed vs single-question evaluation of fine-tuned typed-decisions bundles (the span head is
position-aware). For each bundle and each held-out row, the router battery is asked

  * packed in the trained order            (tool, command, is_write, touches_poe_power)
  * packed in every one of the 24 orders   (accuracy per order, and per slot a question sits in)
  * one question at a time                 ("alone": the question is the whole pack)
  * packed as the production pre-turn hook (tool, is_write) and its reverse

Accuracy is plain gold accuracy per question (the same rule as bench.analyze's `accuracy`,
alternatives not allowed). One JSON receipt holds every bundle.

    PYTHONPATH=$TD python -m bench.packed_vs_single \
        --bundle round1=runs/modernbert-base-desc/bundle \
        --bundle round2=runs/modernbert-base-desc-shuf-e8/bundle \
        --pkg $TD --split-manifest data/td-desc/split_manifest.json --device cuda \
        --out results/packed-vs-single.json
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import platform
import time
from pathlib import Path

from jevlet.backends.encoders import OpenJevBackend
from jevlet.schema import normalize_questions
from bench import netops

TRAINED_ORDER = ["tool", "command", "is_write", "touches_poe_power"]


def gold_index(qid: str, q, row: dict) -> int:
    opts = q.options
    if qid == "tool":
        t = row["expected_tool"] if row["expected_tool"] in opts else netops.NONE
        return opts.index(t)
    if qid == "command":
        c = row.get("expected_command") or ""
        return opts.index(c) if c in opts else opts.index(netops.NONE)
    if qid == "is_write":
        return 1 if row.get("is_write") else 0
    if qid == "touches_poe_power":
        return 1 if row.get("touches_poe_power") else 0
    raise KeyError(qid)


def run_condition(be, rows, qs_by_id, order: list[str], golds: dict) -> dict:
    """Ask the questions in `order` (one pack per row). Returns per-question stats and per-row predictions."""
    stats = {qid: {"n": 0, "correct": 0, "conf_sum": 0.0, "pred_hist": collections.Counter()} for qid in order}
    preds = {qid: [] for qid in order}
    lat = []
    qs = [qs_by_id[qid] for qid in order]
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        ans = be._decide(r["input"], qs)
        lat.append(time.perf_counter() - t0)
        for a in ans:
            p = a.probs
            k = max(range(len(p)), key=lambda j: p[j])
            s = stats[a.qid]
            s["n"] += 1
            s["correct"] += int(k == golds[a.qid][i])
            s["conf_sum"] += float(p[k])
            s["pred_hist"][qs_by_id[a.qid].options[k]] += 1
            preds[a.qid].append(k)
    out = {"order": order, "median_call_ms": sorted(lat)[len(lat) // 2] * 1000.0, "per_question": {}}
    for qid, s in stats.items():
        out["per_question"][qid] = {"n": s["n"], "correct": s["correct"], "acc": s["correct"] / max(1, s["n"]),
                                    "mean_conf": s["conf_sum"] / max(1, s["n"]),
                                    "pred_hist": dict(s["pred_hist"].most_common())}
    out["preds"] = preds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="append", required=True, help="tag=path, repeatable")
    ap.add_argument("--pkg", required=True, help="typed-decisions src/ directory")
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/packed-vs-single.json")
    ap.add_argument("--no-perms", action="store_true", help="skip the 24 orderings")
    a = ap.parse_args()

    test_groups = set(json.load(open(a.split_manifest))["test_groups"])
    rows = [r for r in netops.load_rows() if r["base_input"] in test_groups]
    qs = normalize_questions(netops.router_questions())
    qs_by_id = {q.qid: q for q in qs}
    golds = {qid: [gold_index(qid, qs_by_id[qid], r) for r in rows] for qid in TRAINED_ORDER}

    receipt = {"meta": {"n_rows": len(rows), "split_manifest": a.split_manifest, "trained_order": TRAINED_ORDER,
                        "device": a.device, "host_arch": platform.machine(), "questions": {qid: qs_by_id[qid].options for qid in TRAINED_ORDER},
                        "gold_rule": "plain gold accuracy, alternatives not allowed (bench.analyze `accuracy`)",
                        "row_style": [r.get("style") or "canon" for r in rows], "row_split": [r.get("split") for r in rows]},
               "bundles": {}}
    t_all = time.time()
    for spec in a.bundle:
        tag, _, path = spec.partition("=")
        t0 = time.time()
        be = OpenJevBackend(path, a.device, pkg_dir=a.pkg)
        b = {"path": path, "conditions": {}, "orders": {}, "by_slot": {}}
        b["conditions"]["packed_trained_order"] = run_condition(be, rows, qs_by_id, TRAINED_ORDER, golds)
        for qid in TRAINED_ORDER:
            c = run_condition(be, rows, qs_by_id, [qid], golds)
            b["conditions"][f"alone_{qid}"] = c
        b["conditions"]["packed_pre_turn"] = run_condition(be, rows, qs_by_id, ["tool", "is_write"], golds)
        b["conditions"]["packed_pre_turn_reversed"] = run_condition(be, rows, qs_by_id, ["is_write", "tool"], golds)
        if not a.no_perms:
            slot_tot = {qid: {s: {"correct": 0, "n": 0, "orders": 0} for s in range(1, 5)} for qid in TRAINED_ORDER}
            for perm in itertools.permutations(TRAINED_ORDER):
                order = list(perm)
                c = run_condition(be, rows, qs_by_id, order, golds)
                c.pop("preds")
                b["orders"][",".join(order)] = c
                for s, qid in enumerate(order, start=1):
                    st = slot_tot[qid][s]
                    st["correct"] += c["per_question"][qid]["correct"]
                    st["n"] += c["per_question"][qid]["n"]
                    st["orders"] += 1
            for qid in TRAINED_ORDER:
                accs = [o["per_question"][qid]["acc"] for o in b["orders"].values()]
                b["by_slot"][qid] = {str(s): {**v, "acc": v["correct"] / max(1, v["n"])} for s, v in slot_tot[qid].items()}
                b["by_slot"][qid]["over_orders"] = {"n_orders": len(accs), "acc_mean": sum(accs) / len(accs),
                                                    "acc_min": min(accs), "acc_max": max(accs)}
        b["elapsed_s"] = time.time() - t0
        receipt["bundles"][tag] = b
        print(f"== {tag} ({b['elapsed_s']:.0f}s)")
        for qid in TRAINED_ORDER:
            pk = b["conditions"]["packed_trained_order"]["per_question"][qid]["acc"]
            al = b["conditions"][f"alone_{qid}"]["per_question"][qid]["acc"]
            line = f"  {qid:18s} packed(trained order) {pk:.3f}   alone {al:.3f}"
            if b["by_slot"]:
                o = b["by_slot"][qid]["over_orders"]
                line += f"   24 orders mean {o['acc_mean']:.3f} min {o['acc_min']:.3f} max {o['acc_max']:.3f}   by slot " + \
                        " ".join(f"s{s}={b['by_slot'][qid][str(s)]['acc']:.3f}" for s in range(1, 5))
            print(line)
        pt = b["conditions"]["packed_pre_turn"]["per_question"]
        ptr = b["conditions"]["packed_pre_turn_reversed"]["per_question"]
        print(f"  pre-turn pack (tool,is_write): tool {pt['tool']['acc']:.3f} is_write {pt['is_write']['acc']:.3f} | reversed: tool {ptr['tool']['acc']:.3f} is_write {ptr['is_write']['acc']:.3f}")
    receipt["meta"]["elapsed_s"] = time.time() - t_all
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(receipt, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
