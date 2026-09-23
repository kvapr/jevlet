"""Confirm-card operating points, computed from a gate receipt.

    python -m bench.card_thresholds results/gate-qwen3-4b-q4_0-cpu.json
    -> results/card-thresholds-qwen3-4b-q4_0-cpu.json

Reproduces bench.analyze's gate pipeline (semantic relabel of the real pairs, out-of-fold
temperature scaling with folds grouped by seed request) and then sweeps the card's threshold:
at each value, which of the generative model's real calls would carry a flag (p < t) and which
synthetic negatives would be accepted (p > t). Also records the three-band split at the
lance_local defaults (ok at p >= 0.85, mismatch at p <= 0.4, check between), the risk-coverage
points, a per-proposed-class view, the 20 real confusions with their calibrated p, mock-up
candidates with the exact card sentence produced by annotate_proposal, and the blast-level
distribution per command in the authored evaluation rows (training paraphrases excluded).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np
import yaml

from bench.analyze import _relabel_real
from jevlet.backends.base import Backend
from jevlet.calibrate import cv_temperature
from jevlet.integrations import lance_local as LL
from jevlet.schema import Answer

THRESHOLDS = [0.3, 0.4, 0.5, 0.7, 0.85, 0.9]
OK_ABOVE, MISMATCH_BELOW = 0.85, 0.4          # the annotate_proposal defaults


class FixedP(Backend):
    """A backend that answers every NOUL with one fixed probability; used only to drive
    annotate_proposal so the card sentence comes from the integration code itself."""
    name = "fixed"

    def __init__(self, p: float):
        self.p = float(p)

    def _decide(self, state, questions):
        return [Answer(q.qid, q.kind, [1.0 - self.p, self.p]) for q in questions]


def card_for(x: dict, p: float) -> dict:
    chk = LL.annotate_proposal(FixedP(p), x["input"], x["command"], x["params"])
    return {"level": chk.level, "text": chk.text}


def band(v: float) -> str:
    return "ok" if v >= OK_ABOVE else "mismatch" if v <= MISMATCH_BELOW else "check"


def sweep(p, y, kinds, mk, dang, real):
    rows = []
    for t in THRESHOLDS:
        row = {"threshold": t}
        for k in ("confusion", "diagnose_instead", "format_only", "params", "none"):
            m = real & (mk == k)
            row[f"real_{k}_flagged"] = [int((p[m] < t).sum()), int(m.sum())]
        m = real & (mk == "confusion") & dang
        row["real_dangerous_confusion_flagged"] = [int((p[m] < t).sum()), int(m.sum())]
        m = real & (y == 0)
        row["real_misses_flagged"] = [int((p[m] < t).sum()), int(m.sum())]
        m = real & (y == 1)
        row["real_correct_false_flagged"] = [int((p[m] < t).sum()), int(m.sum())]
        for k in ("gold", "swap_op", "wrong_port"):
            m = kinds == k
            row[f"{k}_accepted"] = [int((p[m] > t).sum()), int(m.sum())]
        m = (kinds == "swap_op") & dang
        row["dangerous_swap_accepted"] = [int((p[m] > t).sum()), int(m.sum())]
        rows.append(row)
    return rows


def blast_by_command():
    rows = []
    for f in sorted(glob.glob("data/netops/authored/*.yaml")):
        if "train_" in os.path.basename(f):
            continue
        d = yaml.safe_load(open(f))
        items = d if isinstance(d, list) else [r for v in d.values() if isinstance(v, list) for r in v]
        rows += [r for r in items if isinstance(r, dict) and "blast" in r]
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        key = r.get("expected_command") or r.get("expected_tool") or "none"
        by[key][int(r["blast"])] += 1
    out = {k: {str(b): n for b, n in sorted(v.items())} for k, v in sorted(by.items())}
    overall = collections.Counter(int(r["blast"]) for r in rows)
    return out, {str(b): n for b, n in sorted(overall.items())}, len(rows)


FIG: str | None = None   # set from --fig: write the risk-coverage figure of the calibrated gate here


def risk_fig(pc, y, dest: str):
    """Risk-coverage curve of the calibrated gate (same construction as bench.analyze), one series,
    annotated at 70% coverage. 8in wide, white background, dpi 150."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conf = np.maximum(pc, 1 - pc)
    correct = (pc > 0.5) == (y == 1)
    order = np.argsort(-conf)
    n = len(order)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(~correct[order]) / np.arange(1, n + 1)
    i70 = int(0.7 * n) - 1
    imax = int(np.argmax(risk))
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150, facecolor="white")
    ax.set_facecolor("white")
    ax.plot(cov, risk, color="#2f6690", linewidth=2)
    ax.plot([cov[i70]], [risk[i70]], "o", color="#2f6690", markersize=7, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"{risk[i70]*100:.1f}% wrong among the most confident 70%", (cov[i70], risk[i70]),
                xytext=(0.34, 0.135), fontsize=9, color="#172026",
                arrowprops={"arrowstyle": "-", "color": "#5e6b72", "linewidth": 0.8})
    ax.plot([1.0], [risk[-1]], "o", color="#2f6690", markersize=7, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"{risk[-1]*100:.1f}% at full coverage", (1.0, risk[-1]), xytext=(0.73, 0.196), fontsize=9,
                color="#172026", ha="left", arrowprops={"arrowstyle": "-", "color": "#5e6b72", "linewidth": 0.8})
    ax.plot([cov[imax]], [risk[imax]], "o", color="#2f6690", markersize=7, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"{risk[imax]*100:.0f}% among the most confident {imax + 1} pairs", (cov[imax], risk[imax]),
                xytext=(0.09, 0.238), fontsize=9, color="#172026",
                arrowprops={"arrowstyle": "-", "color": "#5e6b72", "linewidth": 0.8})
    ax.set_xlim(-0.01, 1.03)
    ax.set_ylim(0, 0.26)
    ax.set_xlabel("coverage: share of pairs the gate decides on, most confident first", fontsize=9, color="#5e6b72")
    ax.set_ylabel("error rate among covered pairs", fontsize=9, color="#5e6b72")
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_yticklabels(["0%", "5%", "10%", "15%", "20%", "25%"], fontsize=8.5, color="#5e6b72")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=8.5, color="#5e6b72")
    ax.grid(axis="y", color="#e4e9e7", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d3dbd8")
    ax.tick_params(colors="#5e6b72", length=0)
    ax.set_title(f"Readout gate, {n} pairs, out-of-fold temperature-scaled", fontsize=10, color="#172026", loc="left")
    fig.tight_layout()
    fig.savefig(dest, facecolor="white")
    plt.close(fig)


def main(path: str):
    d = json.load(open(path))
    res = _relabel_real(d["results"])
    p_raw = np.array([x["p_yes"] for x in res], dtype=float)
    y = np.array([x["label"] for x in res])
    kinds = np.array([x["kind"] for x in res])
    mk = np.array([x.get("miss_kind", "-") for x in res])
    dang = np.array([bool(x.get("dangerous")) for x in res])
    groups = [x["base_input"] for x in res]
    L = np.stack([np.log(np.clip(1 - p_raw, 1e-9, 1)), np.log(np.clip(p_raw, 1e-9, 1))], 1)
    Pc, temps = cv_temperature(L, y, groups, n_folds=5)
    pc = Pc[:, 1]
    real = kinds == "real"

    out = {"source": os.path.basename(path), "backend": d["meta"].get("backend"), "n": len(res),
           "n_real": int(real.sum()), "temperatures": temps, "thresholds": THRESHOLDS,
           "defaults": {"ok_above": OK_ABOVE, "mismatch_below": MISMATCH_BELOW},
           "table_calibrated": sweep(pc, y, kinds, mk, dang, real),
           "table_raw": sweep(p_raw, y, kinds, mk, dang, real)}

    # three-band split at the code's defaults, calibrated p
    bands = {}
    for k in ("confusion", "diagnose_instead", "format_only", "params", "none"):
        m = real & (mk == k)
        bands[f"real_{k}"] = dict(collections.Counter(band(v) for v in pc[m]))
    for k in ("gold", "swap_op", "wrong_port"):
        bands[k] = dict(collections.Counter(band(v) for v in pc[kinds == k]))
    bands["dangerous_swap"] = dict(collections.Counter(band(v) for v in pc[(kinds == "swap_op") & dang]))
    bands["real_dangerous_confusion"] = dict(collections.Counter(band(v) for v in pc[real & (mk == "confusion") & dang]))
    out["bands_at_defaults"] = bands
    out["real_mean_p_cal"] = {"correct": float(pc[real & (y == 1)].mean()), "wrong": float(pc[real & (y == 0)].mean())}

    # risk-coverage points (same construction as bench.analyze._risk_coverage_fig)
    conf = np.maximum(pc, 1 - pc)
    correct = (pc > 0.5) == (y == 1)
    order = np.argsort(-conf)
    n = len(order)
    risk = np.cumsum(~correct[order]) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    imax = int(np.argmax(risk[n // 10:]) + n // 10)    # ignore the first 10% where one error dominates
    out["risk_coverage"] = {"risk_at_50pct": float(risk[int(0.5 * n) - 1]), "risk_at_70pct": float(risk[int(0.7 * n) - 1]),
                            "risk_at_90pct": float(risk[int(0.9 * n) - 1]), "risk_at_full": float(risk[-1]),
                            "risk_max_after_10pct": float(risk[imax]), "coverage_at_max": float(cov[imax])}

    # per proposed class on the real pairs
    cls_rows = collections.defaultdict(lambda: {"n": 0, "wrong": 0, "wrong_flagged_0.4": 0, "wrong_flagged_0.5": 0,
                                                "correct_false_flagged_0.4": 0, "correct_false_flagged_0.5": 0})
    for i in np.where(real)[0]:
        x = res[i]
        cls = LL.OP_CLASS.get(x["command"], x["command"])
        r = cls_rows[cls]
        r["n"] += 1
        if y[i] == 0:
            r["wrong"] += 1
            r["wrong_flagged_0.4"] += int(pc[i] < 0.4)
            r["wrong_flagged_0.5"] += int(pc[i] < 0.5)
        else:
            r["correct_false_flagged_0.4"] += int(pc[i] < 0.4)
            r["correct_false_flagged_0.5"] += int(pc[i] < 0.5)
    out["real_by_proposed_class"] = dict(sorted(cls_rows.items(), key=lambda kv: -kv[1]["n"]))

    # the 20 real confusions
    conf_rows = []
    for i in np.where(real & (mk == "confusion"))[0]:
        x = res[i]
        conf_rows.append({"input": x["input"], "true_command": x["true_command"], "command": x["command"],
                          "params": x["params"], "dangerous": bool(dang[i]), "p_cal": float(pc[i]), "p_raw": float(p_raw[i]),
                          "band": band(pc[i])})
    out["real_confusions"] = sorted(conf_rows, key=lambda r: r["p_cal"])

    # mock-up candidates with the exact card sentence from the integration code
    cands = {"ok": [], "check": [], "mismatch": []}
    for i in np.where(real)[0]:
        x = res[i]
        if x["command"] not in LL.OP_CLASS:
            continue
        b = band(pc[i])
        cands[b].append({"input": x["input"], "true_command": x["true_command"], "command": x["command"],
                         "params": x["params"], "label": int(y[i]), "miss_kind": str(mk[i]), "dangerous": bool(dang[i]),
                         "p_cal": float(pc[i]), "card": card_for(x, float(pc[i]))})
    cands["ok"] = sorted(cands["ok"], key=lambda r: -r["p_cal"])[:8]
    cands["check"] = sorted(cands["check"], key=lambda r: -r["p_cal"])[:10]
    cands["mismatch"] = sorted([r for r in cands["mismatch"] if r["label"] == 0], key=lambda r: r["p_cal"])
    out["mockup_candidates"] = cands

    out["blast_by_command_authored"], out["blast_overall_authored"], out["blast_rows_authored"] = blast_by_command()
    out["class_text"] = LL.CLASS_TEXT
    out["op_class"] = LL.OP_CLASS

    if FIG:
        risk_fig(pc, y, FIG)
    dest = f"results/card-thresholds-{os.path.basename(path)[5:]}"
    json.dump(out, open(dest, "w"), indent=1)

    print(f"{out['source']}: n={out['n']} real={out['n_real']} T={[round(t, 2) for t in temps]}")
    print("| t | confusions flagged | dangerous conf. | diagnose flagged | format slips flagged | correct false-flagged | gold accepted | swap accepted | dangerous swap accepted | wrong port accepted |")
    for r in out["table_calibrated"]:
        f = lambda k: f"{r[k][0]} / {r[k][1]}"
        print(f"| {r['threshold']} | {f('real_confusion_flagged')} | {f('real_dangerous_confusion_flagged')} | {f('real_diagnose_instead_flagged')} | "
              f"{f('real_format_only_flagged')} | {f('real_correct_false_flagged')} | {f('gold_accepted')} | {f('swap_op_accepted')} | "
              f"{f('dangerous_swap_accepted')} | {f('wrong_port_accepted')} |")
    print("bands at defaults:", json.dumps(bands))
    print("risk-coverage:", json.dumps(out["risk_coverage"]))
    print("mean p_cal real correct / wrong:", out["real_mean_p_cal"])
    print("by proposed class:", json.dumps(out["real_by_proposed_class"], indent=0))
    print("confusions:")
    for r in out["real_confusions"]:
        print(f"  p={r['p_cal']:.2f} {'D' if r['dangerous'] else ' '} {r['true_command']} -> {r['command']} {r['params']} | {r['input']}")
    for b, lst in cands.items():
        print(f"== {b} candidates")
        for r in lst:
            print(f"  p={r['p_cal']:.3f} label={r['label']} {r['miss_kind']} {r['true_command']} -> {r['command']} {r['params']} | {r['input']}")
            print(f"     {r['card']['text']}")
    print("blast by command:", json.dumps(out["blast_by_command_authored"]))
    print("blast overall:", out["blast_overall_authored"], "rows", out["blast_rows_authored"])
    print("wrote", dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default="results/gate-qwen3-4b-q4_0-cpu.json")
    ap.add_argument("--fig", default=None, help="also write the risk-coverage PNG of the calibrated gate to this path")
    a = ap.parse_args()
    FIG = a.fig
    main(a.file)
