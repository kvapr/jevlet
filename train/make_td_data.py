"""Convert the netops rows into typed-decisions JSONL (state + typed questions with gold indices)
for fine-tuning an encoder decision model with the kotoba-lang/typed-decisions trainer.

Splits are GROUPED by base_input: every paraphrase (evaluation styles A/B/C and the training
paraphrases) of one seed request lands in the same split. Test = the held-out groups; the
authored evaluation rows (adversarial confusables, other intents) are split the same way.

    python -m train.make_td_data --out data/td --test-frac 0.25 --seed 0

Writes train.jsonl / val.jsonl / test.jsonl and split_manifest.json (which base_inputs went where).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
from pathlib import Path

import yaml

from bench import netops

ROOT = Path(__file__).resolve().parents[1]


def _q(qid, kind, instructions, options, gold, descriptions=None):
    if descriptions:
        options = [f"{o}: {descriptions[o]}" if descriptions.get(o) else o for o in options]
    return {"qid": qid, "kind": kind, "instructions": instructions, "options": options, "gold": int(gold)}


def router_examples(row: dict, cat: dict, tool_opts: list[str], cmd_opts: list[str], with_desc=True) -> dict:
    """One typed-decisions Example (state = request text) carrying the router battery.
    Option texts are rendered exactly as jevlet.backends.encoders renders them at inference
    ("<option>: <description>"), so a trained bundle sees the same strings."""
    from jevlet.schema import normalize_questions
    qn = {q.qid: q for q in normalize_questions(netops.router_questions(cat, netops.CATALOG_CMDS))}
    tool = row["expected_tool"] if row["expected_tool"] in tool_opts else netops.NONE
    cmd = row["expected_command"] if row["expected_command"] in cmd_opts else netops.NONE
    tool_desc = qn["tool"].descriptions if with_desc else None
    cmd_desc = qn["command"].descriptions if with_desc else None
    qs = [
        _q("tool", "choice", qn["tool"].instructions, tool_opts, tool_opts.index(tool), tool_desc),
        _q("command", "choice", qn["command"].instructions, cmd_opts, cmd_opts.index(cmd), cmd_desc),
        _q("is_write", "noul", qn["is_write"].instructions, ["no", "yes"], 1 if row["is_write"] else 0),
        _q("touches_poe_power", "noul", qn["touches_poe_power"].instructions, ["no", "yes"], 1 if row["touches_poe_power"] else 0),
    ]
    return {"state": row["input"], "source": f"netops/{row.get('split','?')}",
            "meta": {"base_input": row["base_input"], "style": row.get("style", ""), "ambiguity": row.get("ambiguity", 0),
                     "dangerous_pair": row.get("dangerous_pair", "none")},
            "questions": qs}


def gate_example(pair: dict) -> dict:
    gq = netops.gate_questions()["matches_request"]
    return {"state": netops.gate_state(pair["input"], pair["command"], pair["params"]),
            "source": f"netops-gate/{pair['kind']}",
            "meta": {"base_input": pair["base_input"], "style": pair.get("style", ""), "kind": pair["kind"],
                     "dangerous": pair.get("dangerous", False)},
            "questions": [_q("matches_request", "noul", gq["instructions"], ["no", "yes"], pair["label"])]}


def load_train_paraphrases() -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(netops.OUT / "authored" / "train_*.yaml"))):
        rows += yaml.safe_load(open(p)) or []
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/td")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true", help="also emit confirm-gate examples (synthetic pairs from train groups)")
    ap.add_argument("--no-desc", action="store_true", help="bare option names (fits 512-token encoders)")
    ap.add_argument("--shuffle-questions", type=int, default=0,
                    help="emit K extra copies of each TRAIN example with the question order permuted (the span head is position-aware)")
    ap.add_argument("--gate-only", action="store_true",
                    help="emit ONLY confirm-gate examples, class-balanced (positives repeated to match negatives)")
    a = ap.parse_args()
    global router_examples
    if a.no_desc:
        _re = router_examples
        router_examples = lambda row, cat, t, c: _re(row, cat, t, c, with_desc=False)
    os.makedirs(a.out, exist_ok=True)
    cat = netops.catalog()
    tool_opts = list(netops.TOOL_OPTIONS.keys())
    cmd_opts = netops.CATALOG_CMDS + [netops.NONE]

    rows = netops.load_rows()                       # canonical + paraphrase + authored eval rows
    by_base = {}
    for r in rows:
        by_base.setdefault(r["base_input"], {"eval": [], "train": []})["eval"].append(r)
    for p in load_train_paraphrases():              # training-only paraphrases attach to their seed group
        seed_row = next((r for r in rows if r["input"] == p["base_input"]), None)
        if seed_row is None:
            continue
        r = dict(seed_row)
        r["input"], r["style"], r["split"] = p["input"], "train-" + p.get("style", "?"), "train_paraphrase"
        by_base.setdefault(p["base_input"], {"eval": [], "train": []})["train"].append(r)

    groups = sorted(by_base)
    rng = random.Random(a.seed)
    rng.shuffle(groups)
    n_test = int(round(len(groups) * a.test_frac))
    n_val = int(round(len(groups) * a.val_frac))
    test_g, val_g, train_g = set(groups[:n_test]), set(groups[n_test:n_test + n_val]), set(groups[n_test + n_val:])

    train, val, test = [], [], []
    for g, d in by_base.items():
        if a.gate_only:
            break
        target = test if g in test_g else val if g in val_g else train
        for r in d["eval"]:
            target.append(router_examples(r, cat, tool_opts, cmd_opts))
        if g not in test_g:                          # training paraphrases never touch the test groups
            for r in d["train"]:
                target.append(router_examples(r, cat, tool_opts, cmd_opts))
    if a.gate_only:
        train_rows = [r for g in train_g for r in by_base[g]["eval"] + by_base[g]["train"] if r.get("expected_command")]
        val_rows = [r for g in val_g for r in by_base[g]["eval"] if r.get("expected_command")]
        test_rows = [r for g in test_g for r in by_base[g]["eval"] if r.get("expected_command")]
        for rows_, target in ((train_rows, train), (val_rows, val), (test_rows, test)):
            pairs = netops.gate_pairs(None, rows=rows_, synthetic=True)
            pos = [p for p in pairs if p["label"] == 1]
            neg = [p for p in pairs if p["label"] == 0]
            if target is train and pos and len(neg) > len(pos):   # balance the training set
                k = len(neg) // len(pos)
                pos = pos * k + pos[: len(neg) - k * len(pos)]
            for pr in pos + neg:
                target.append(gate_example(pr))
    if a.gate:
        train_rows = [r for g in train_g | val_g for r in by_base[g]["eval"] + by_base[g]["train"]]
        test_rows = [r for g in test_g for r in by_base[g]["eval"]]
        for pr in netops.gate_pairs(None, rows=[r for r in train_rows if r.get("expected_command")], synthetic=True):
            (val if pr["base_input"] in val_g else train).append(gate_example(pr))
        for pr in netops.gate_pairs(None, rows=[r for r in test_rows if r.get("expected_command")], synthetic=True):
            test.append(gate_example(pr))

    if a.shuffle_questions:
        extra = []
        for e in train:
            for _ in range(a.shuffle_questions):
                qs = list(e["questions"])
                rng.shuffle(qs)
                extra.append({**e, "questions": qs, "meta": {**e["meta"], "qshuffled": True}})
        train += extra
    rng.shuffle(train)
    for name, ex in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for e in ex:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    json.dump({"seed": a.seed, "test_groups": sorted(test_g), "val_groups": sorted(val_g),
               "n_train": len(train), "n_val": len(val), "n_test": len(test)},
              open(os.path.join(a.out, "split_manifest.json"), "w"), indent=1)
    print(f"groups: {len(groups)} (test {len(test_g)}, val {len(val_g)}, train {len(train_g)}); "
          f"examples train {len(train)} val {len(val)} test {len(test)} -> {a.out}")


if __name__ == "__main__":
    main()
