"""Count the pack shapes a typed-decisions training set shows the span head: how many questions
ride on each example, in which orders, and where the command question sits. Written as a receipt
next to results/packed-vs-single.json because the position-awareness of the head is a property
of exactly these counts.

    python -m bench.pack_shapes data/td-desc data/td-desc-shuf --out results/pack-shapes.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def count(path: Path) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        f = path / f"{split}.jsonl"
        if not f.exists():
            continue
        n = 0
        n_questions = collections.Counter()
        orders = collections.Counter()
        command_slot = collections.Counter()
        sources = collections.Counter()
        qshuffled = 0
        single_gold_no = 0
        single_total = 0
        for line in open(f):
            e = json.loads(line)
            n += 1
            qids = [q["qid"] for q in e["questions"]]
            n_questions[len(qids)] += 1
            sources[e.get("source", "?").split("/")[0]] += 1
            if e.get("meta", {}).get("qshuffled"):
                qshuffled += 1
            if "command" in qids:
                command_slot[qids.index("command") + 1] += 1
            if len(qids) > 1:
                orders[",".join(qids)] += 1
            else:
                single_total += 1
                single_gold_no += int(e["questions"][0]["gold"] == 0)
        out[split] = {"examples": n, "questions_per_example": dict(sorted(n_questions.items())),
                      "sources": dict(sources), "qshuffled": qshuffled,
                      "command_slot": dict(sorted(command_slot.items())),
                      "distinct_orders": len(orders), "orders": dict(orders.most_common()),
                      "single_question_examples": single_total, "single_question_gold_no": single_gold_no}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/pack-shapes.json")
    a = ap.parse_args()
    rep = {d: count(Path(d)) for d in a.dirs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    for d, r in rep.items():
        for split, s in r.items():
            print(d, split, s["examples"], "examples; q/ex", s["questions_per_example"], "; orders", s["distinct_orders"],
                  "; command slot", s["command_slot"], "; single-q", s["single_question_examples"], "gold no", s["single_question_gold_no"])
    print("wrote", a.out)
