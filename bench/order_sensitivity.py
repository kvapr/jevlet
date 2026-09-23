"""Order sensitivity of the next-token readout: what flips between option rotations, what
rotation averaging buys, and where the position bias sits.

Reads the router-*-rot3.json receipts written by bench.run_readout. For rotation r the option
list is rotated cyclically so that canonical option r is shown FIRST (label A) and canonical
option (p + r) mod k sits at position p; `per_rotation[r]` holds that rotation's probabilities
mapped back to CANONICAL option order (jevlet/backends/llamacpp.py::readout_rotation). `probs`
is the arithmetic mean of the rotations, `logits` the mean log-probability (combine_rotations).
NOUL questions are recorded with one rotation, so no flip statistic exists for them.

    python -m bench.order_sensitivity --figdir docs/artifact
      -> results/order-sensitivity.json, results/summary/order-sensitivity.md,
         <figdir>/order-sensitivity-{flip-rate,averaging,position-bias}.png
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import itertools
import json
import os
from pathlib import Path

import numpy as np

from jevlet import metrics as M
from bench.netops import NONE, DANGEROUS_PAIRS

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

FILES = [
    ("results/router-qwen3-4b-q4_0-cpu-rot3.json", "Qwen3-4B", "260 base rows"),
    ("results/router-qwen3-4b-q4_0-cpu-authored-rot3.json", "Qwen3-4B", "259 authored rows"),
    ("results/router-qwen3-4b-q4_0-cpu-all-rot3.json", "Qwen3-4B", "519 merged rows"),
    ("results/router-qwen3.5-0.8b-q4_0-cpu-v2-rot3.json", "Qwen3.5-0.8B (thinking off)", "260 base rows"),
]

DEFINITIONS = {
    "flip_rate": "share of rows whose argmax differs between at least two of the R rotations",
    "pairwise_flip_rate": "share of rows whose argmax differs between rotations r and s, per pair (r, s), and the mean over pairs",
    "acc_single_rotation": "accuracy of the argmax of rotation r alone",
    "acc_averaged": "accuracy of the argmax of the arithmetic mean of the R rotations (the `probs` field)",
    "acc_majority": "accuracy of the plurality vote over the R argmaxes; a three-way tie falls back to the averaged argmax",
    "acc_oracle_any": "share of rows where at least one rotation's argmax is the gold (an upper bound, not a method)",
    "acc_all_rotations": "share of rows where every rotation's argmax is the gold",
    "confident_flips": "flipped rows on which at least one rotation put max p > 0.9 on its answer",
    "confident_both_ways": "flipped rows on which two rotations each put max p > 0.9 on different answers",
    "first_shown": "option shown first (label A) in rotation r is canonical option r; win = that option is the rotation's argmax",
    "position_shares": "pooled over rotations x rows: share of wins at display position p versus share of gold labels at p",
    "paired_first_vs_elsewhere": "for the R options that are shown first in some rotation: mean probability and win rate when first versus in the other rotations, same rows",
    "geometric_vs_arithmetic": "rows where argmax of the mean log-probability (used for temperature scaling) differs from argmax of the mean probability",
}


def _labels_for(qid, q, rows):
    opts = q["options"]
    y = []
    for r in rows:
        if qid == "tool":
            t = r["expected_tool"] if r["expected_tool"] in opts else NONE
            y.append(opts.index(t))
        elif qid == "command":
            c = r.get("expected_command") or ""
            y.append(opts.index(c) if c in opts else opts.index(NONE))
        elif qid == "is_write":
            y.append(1 if r.get("is_write") else 0)
        elif qid == "touches_poe_power":
            y.append(1 if r.get("touches_poe_power") else 0)
        else:
            y.append(-1)
    return np.array(y)


def _majority(A, Pavg):
    R, N = A.shape
    out = np.empty(N, dtype=int)
    for i in range(N):
        c = collections.Counter(A[:, i].tolist()).most_common()
        out[i] = c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) else int(Pavg[i].argmax())
    return out


def analyze_question(qid, q, rows, recs):
    opts = q["options"]
    k = len(opts)
    recs = sorted(recs, key=lambda x: x["i"])
    have = [x for x in recs if x.get("per_rotation")]
    y_all = _labels_for(qid, q, rows)
    out = {"kind": q["kind"], "k": k, "n_rows": len(recs), "rows_with_per_rotation": len(have)}
    if not have:
        Pavg = np.array([x["probs"] for x in recs], dtype=float)
        out["acc_averaged"] = float((Pavg.argmax(1) == y_all).mean())
        out["note"] = "per_rotation not recorded in this receipt; averaged probabilities only"
        return out
    R = len(have[0]["per_rotation"])
    out["rotations"] = R
    idx = np.array([x["i"] for x in have])
    y = y_all[idx]
    sub = [rows[i] for i in idx]
    P = np.array([x["per_rotation"] for x in have], dtype=float)          # (N, R, k)
    Pavg = np.array([x["probs"] for x in have], dtype=float)               # (N, k)
    L = np.array([x["logits"] for x in have], dtype=float)                 # (N, k) mean log p
    out["max_abs_diff_probs_vs_mean_of_rotations"] = float(np.abs(P.mean(1) - Pavg).max())
    out["acc_averaged"] = float((Pavg.argmax(1) == y).mean())
    if qid == "command":
        alt = np.array([opts[p] in set(r.get("alt_ok") or []) for r, p in zip(sub, Pavg.argmax(1))])
        out["acc_averaged_allowing_alt_ok"] = float(((Pavg.argmax(1) == y) | alt).mean())
    if R < 2:
        out["note"] = "one rotation recorded (NOUL questions are not rotated); flips not measurable"
        out["acc_single_rotation"] = [float((P[:, 0].argmax(1) == y).mean())]
        return out
    A = P.argmax(2).T                                                      # (R, N)
    N = len(y)
    flip = np.array([len(set(A[:, i].tolist())) > 1 for i in range(N)])
    out["flip_rate"] = float(flip.mean())
    out["n_flipped"] = int(flip.sum())
    pw = {f"{r}-{s}": float((A[r] != A[s]).mean()) for r, s in itertools.combinations(range(R), 2)}
    out["pairwise_flip_rate"] = pw
    out["pairwise_flip_rate_mean"] = float(np.mean(list(pw.values())))
    out["acc_single_rotation"] = [float((A[r] == y).mean()) for r in range(R)]
    out["acc_single_rotation_mean"] = float(np.mean(out["acc_single_rotation"]))
    maj = _majority(A, Pavg)
    out["acc_majority"] = float((maj == y).mean())
    out["acc_oracle_any"] = float(np.any(A == y[None, :], axis=0).mean())
    out["acc_all_rotations"] = float(np.all(A == y[None, :], axis=0).mean())
    geo = L.argmax(1)
    out["geometric_vs_arithmetic"] = {"n_rows_differ": int((geo != Pavg.argmax(1)).sum()),
                                      "acc_geometric": float((geo == y).mean())}
    # confidence and flips
    maxp = P.max(2)                                                        # (N, R)
    conf_any = (maxp > 0.9).any(1)
    out["confident_flips"] = int((flip & conf_any).sum())
    both = np.zeros(N, dtype=bool)
    for i in range(N):
        winners = {int(A[r, i]) for r in range(R) if maxp[i, r] > 0.9}
        both[i] = len(winners) > 1
    out["confident_both_ways"] = int(both.sum())
    out["rows_with_any_rotation_above_0.9"] = int(conf_any.sum())
    wrong_conf = [int(((A[r] != y) & (maxp[:, r] > 0.9)).sum()) for r in range(R)]
    out["wrong_above_0.9_single_rotation"] = wrong_conf
    rescued = [int(((A[r] != y) & (maxp[:, r] > 0.9) & (Pavg.argmax(1) == y)).sum()) for r in range(R)]
    out["wrong_above_0.9_single_rotation_rescued_by_averaging"] = rescued
    out["wrong_above_0.9_averaged"] = int(((Pavg.argmax(1) != y) & (Pavg.max(1) > 0.9)).sum())
    # flips as an uncertainty signal
    correct = Pavg.argmax(1) == y
    out["acc_averaged_on_flipped_rows"] = float(correct[flip].mean()) if flip.any() else None
    out["acc_averaged_on_stable_rows"] = float(correct[~flip].mean()) if (~flip).any() else None
    out["flip_rate_given_averaged_wrong"] = float(flip[~correct].mean()) if (~correct).any() else None
    out["flip_rate_given_averaged_right"] = float(flip[correct].mean()) if correct.any() else None
    out["mean_maxp_averaged_flipped"] = float(Pavg.max(1)[flip].mean()) if flip.any() else None
    out["mean_maxp_averaged_stable"] = float(Pavg.max(1)[~flip].mean()) if (~flip).any() else None
    out["auroc_stable_predicts_correct"] = M.auroc((~flip).astype(float), correct.astype(int))
    out["auroc_maxp_predicts_correct"] = M.auroc(Pavg.max(1), correct.astype(int))
    out["abstain_on_flips"] = {"coverage": float((~flip).mean()),
                               "acc_on_covered": float(correct[~flip].mean()) if (~flip).any() else None}
    # what flips: option pairs
    pairs = collections.Counter()
    none_i = opts.index(NONE) if NONE in opts else -1
    n_none = n_dang = 0
    for i in np.where(flip)[0]:
        w = sorted(set(A[:, i].tolist()))
        if none_i in w:
            n_none += 1
        if any({opts[a], opts[b]} in DANGEROUS_PAIRS for a, b in itertools.combinations(w, 2)):
            n_dang += 1
        for a, b in itertools.combinations(w, 2):
            pairs[(opts[a], opts[b])] += 1
    out["flipped_rows_involving_none_of_these"] = n_none
    out["flipped_rows_involving_a_dangerous_pair"] = n_dang
    out["top_flip_pairs"] = [{"a": a, "b": b, "n": n} for (a, b), n in pairs.most_common(10)]
    # which rotation was right on flipped rows where the average is wrong
    out["flipped_and_averaged_wrong_but_some_rotation_right"] = int((flip & ~correct & np.any(A == y[None, :], 0)).sum())
    # position bias
    first = {}
    for r in range(R):
        first[str(r)] = {"option": opts[r], "win_rate": float((A[r] == r).mean()), "gold_share": float((y == r).mean()),
                         "mean_p_when_first": float(P[:, r, r].mean()),
                         "mean_p_elsewhere": float(np.mean([P[:, s, r].mean() for s in range(R) if s != r])),
                         "win_rate_elsewhere": float(np.mean([(A[s] == r).mean() for s in range(R) if s != r]))}
    out["first_shown"] = first
    out["first_shown_pooled"] = {"win_rate": float(np.mean([v["win_rate"] for v in first.values()])),
                                 "gold_share": float(np.mean([v["gold_share"] for v in first.values()])),
                                 "chance": 1.0 / k,
                                 "mean_p_when_first": float(np.mean([v["mean_p_when_first"] for v in first.values()])),
                                 "mean_p_elsewhere": float(np.mean([v["mean_p_elsewhere"] for v in first.values()])),
                                 "win_rate_elsewhere": float(np.mean([v["win_rate_elsewhere"] for v in first.values()]))}
    win_pos = np.zeros(k)
    gold_pos = np.zeros(k)
    for r in range(R):
        for i in range(N):
            win_pos[(A[r, i] - r) % k] += 1
            gold_pos[(y[i] - r) % k] += 1
    win_pos /= R * N
    gold_pos /= R * N
    out["position_shares"] = {"win": [float(v) for v in win_pos], "gold": [float(v) for v in gold_pos],
                              "labels": list(LETTERS[:k]),
                              "first": {"win": float(win_pos[0]), "gold": float(gold_pos[0])},
                              "last": {"win": float(win_pos[-1]), "gold": float(gold_pos[-1])},
                              "most_won": {"position": int(win_pos.argmax()), "label": LETTERS[int(win_pos.argmax())],
                                           "win": float(win_pos.max()), "gold": float(gold_pos[win_pos.argmax()])},
                              "max_abs_gap": float(np.abs(win_pos - gold_pos).max()),
                              "total_variation": float(0.5 * np.abs(win_pos - gold_pos).sum())}
    # breakdowns
    styles = np.array([r.get("style") or "canon" for r in sub])
    splits = np.array([r.get("split") or "?" for r in sub])
    def brk(mask):
        return {"n": int(mask.sum()), "flip_rate": float(flip[mask].mean()), "acc_averaged": float(correct[mask].mean()),
                "acc_single_rotation_mean": float(np.mean([(A[r][mask] == y[mask]).mean() for r in range(R)])),
                "confident_flips": int((flip & conf_any)[mask].sum())}
    out["by_style"] = {s: brk(styles == s) for s in sorted(set(styles))}
    out["by_split"] = {s: brk(splits == s) for s in sorted(set(splits))}
    return out


def analyze_file(path, model, rows_label):
    d = json.load(open(path))
    meta, res = d["meta"], d["results"]
    rows = meta["rows"]
    out = {"file": os.path.basename(path), "model": model, "rows": rows_label, "backend": meta.get("backend"),
           "rotations_requested": meta.get("rotations"), "n": len(rows), "questions": {}}
    for qid, q in meta["questions"].items():
        out["questions"][qid] = analyze_question(qid, q, rows, res[qid])
    return out


# ------------------------------------------------------------------ report
def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(summary):
    lines = ["# Order sensitivity of the readout (bench.order_sensitivity)", "",
             f"Generated {summary['generated']}. Definitions are in results/order-sensitivity.json.", "",
             "| model | rows | question | k | R | flip rate | pairwise mean | single rotation r0 / r1 / r2 | averaged | majority | any rotation right | confident flips | first-shown wins vs gold share (chance) |",
             "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for f in summary["files"]:
        for qid, r in f["questions"].items():
            if "flip_rate" not in r:
                note = r.get("note", "")
                lines.append(f"| {f['model']} | {f['rows']} | {qid} | {r['k']} | {r.get('rotations', '-')} | n/a | n/a | n/a | {_pct(r['acc_averaged'])} | n/a | n/a | n/a | {note} |")
                continue
            fs = r["first_shown_pooled"]
            singles = " / ".join(f"{100 * a:.1f}" for a in r["acc_single_rotation"])
            lines.append(f"| {f['model']} | {f['rows']} | {qid} | {r['k']} | {r['rotations']} | {_pct(r['flip_rate'])} | {_pct(r['pairwise_flip_rate_mean'])} | {singles} | "
                         f"{_pct(r['acc_averaged'])} | {_pct(r['acc_majority'])} | {_pct(r['acc_oracle_any'])} | {r['confident_flips']} / {r['n_flipped']} | "
                         f"{_pct(fs['win_rate'])} vs {_pct(fs['gold_share'])} ({_pct(fs['chance'])}) |")
    lines += ["", "## Breakdowns (choice questions with R = 3)", "",
              "| model | rows | question | group | n | flip rate | averaged acc | single-rotation acc (mean) | confident flips |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for f in summary["files"]:
        for qid, r in f["questions"].items():
            if "by_split" not in r:
                continue
            for name, groups in (("style", r["by_style"]), ("split", r["by_split"])):
                for g, v in groups.items():
                    lines.append(f"| {f['model']} | {f['rows']} | {qid} | {name}={g} | {v['n']} | {_pct(v['flip_rate'])} | {_pct(v['acc_averaged'])} | {_pct(v['acc_single_rotation_mean'])} | {v['confident_flips']} |")
    lines += ["", "## Flips as an uncertainty signal", "",
              "| model | rows | question | averaged acc on flipped rows | on stable rows | P(flip given wrong) | P(flip given right) | abstain on flips: coverage, acc | AUROC stable→correct | AUROC max p→correct | wrong > 0.9 per rotation | rescued by averaging | wrong > 0.9 after averaging |",
              "|---|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---:|"]
    for f in summary["files"]:
        for qid, r in f["questions"].items():
            if "flip_rate" not in r:
                continue
            ab = r["abstain_on_flips"]
            lines.append(f"| {f['model']} | {f['rows']} | {qid} | {_pct(r['acc_averaged_on_flipped_rows'])} | {_pct(r['acc_averaged_on_stable_rows'])} | "
                         f"{_pct(r['flip_rate_given_averaged_wrong'])} | {_pct(r['flip_rate_given_averaged_right'])} | {_pct(ab['coverage'])}, {_pct(ab['acc_on_covered'])} | "
                         f"{r['auroc_stable_predicts_correct']:.3f} | {r['auroc_maxp_predicts_correct']:.3f} | {r['wrong_above_0.9_single_rotation']} | "
                         f"{r['wrong_above_0.9_single_rotation_rescued_by_averaging']} | {r['wrong_above_0.9_averaged']} |")
    lines += ["", "## Position shares (pooled over rotations x rows)", "",
              "| model | rows | question | first position: won / gold | last position: won / gold | most-won position | max gap | total variation |",
              "|---|---|---|---|---|---|---:|---:|"]
    for f in summary["files"]:
        for qid, r in f["questions"].items():
            if "position_shares" not in r:
                continue
            ps = r["position_shares"]
            mw = ps["most_won"]
            lines.append(f"| {f['model']} | {f['rows']} | {qid} | {_pct(ps['first']['win'])} / {_pct(ps['first']['gold'])} | {_pct(ps['last']['win'])} / {_pct(ps['last']['gold'])} | "
                         f"{mw['label']}: {_pct(mw['win'])} won, {_pct(mw['gold'])} gold | {_pct(ps['max_abs_gap'])} | {_pct(ps['total_variation'])} |")
    lines += ["", "## Top flip pairs", ""]
    for f in summary["files"]:
        for qid, r in f["questions"].items():
            if "top_flip_pairs" not in r:
                continue
            tp = "; ".join(f"{p['a']} ↔ {p['b']} ({p['n']})" for p in r["top_flip_pairs"][:6])
            lines.append(f"- {f['model']}, {f['rows']}, {qid}: {tp}. Flipped rows involving none_of_these: {r['flipped_rows_involving_none_of_these']}; involving a dangerous pair: {r['flipped_rows_involving_a_dangerous_pair']}.")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ figures
# One readout hue for both models (rows are labeled by model, so hue does not carry identity); the
# page's warm token marks the gold; a light gray de-emphasizes single rotations. Pairs validated with
# the dataviz palette checker on a white surface (blue/warm: CVD dE 23.4, normal 27.9).
COL = {"blue": "#3b6ea5", "gold": "#c98a2b", "gray": "#8a9499", "range": "#e6e9e8", "grid": "#e1e0d9", "ink": "#172026", "muted": "#5e6b72"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": COL["ink"], "xtick.color": COL["muted"],
                         "ytick.color": COL["muted"], "text.color": COL["ink"], "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def _panels(summary):
    """The four (model, question) combinations that carry rotation data, in display order."""
    want = [("router-qwen3-4b-q4_0-cpu-authored-rot3.json", "tool", "Qwen3-4B"), ("router-qwen3-4b-q4_0-cpu-authored-rot3.json", "command", "Qwen3-4B"),
            ("router-qwen3.5-0.8b-q4_0-cpu-v2-rot3.json", "tool", "Qwen3.5-0.8B"), ("router-qwen3.5-0.8b-q4_0-cpu-v2-rot3.json", "command", "Qwen3.5-0.8B")]
    by_file = {f["file"]: f for f in summary["files"]}
    out = []
    for fn, qid, model in want:
        f = by_file[fn]
        r = f["questions"][qid]
        out.append((f"{model} · {qid} ({r['k']}-way)\n{f['rows']}", r))
    return out


def fig_flip_rate(summary, figdir):
    plt = _plt()
    from matplotlib.patches import Patch
    panels = _panels(summary)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ys = np.arange(len(panels))[::-1]
    for yy, (label, r) in zip(ys, panels):
        fr, cf = 100 * r["flip_rate"], 100 * r["confident_flips"] / r["n_rows"]
        ax.barh(yy, fr, height=0.52, color=COL["blue"], alpha=0.35, edgecolor="none")
        ax.barh(yy, cf, height=0.52, color=COL["blue"], edgecolor="white", linewidth=1)
        txt = f"{fr:.0f}%  ·  {r['n_flipped']} of {r['n_rows']} rows  ·  {r['confident_flips']} confident"
        if fr > 60:
            ax.text(fr - 1.5, yy, txt, va="center", ha="right", fontsize=8, color=COL["ink"])
        else:
            ax.text(fr + 1.5, yy, txt, va="center", ha="left", fontsize=8, color=COL["ink"])
    ax.set_yticks(ys)
    ax.set_yticklabels([p[0] for p in panels], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(panels) - 0.4)
    ax.set_xlabel("share of rows whose argmax differs between at least two of the 3 rotations, %")
    ax.xaxis.grid(True, color=COL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=COL["blue"], alpha=0.35, label="rows that flip"),
                       Patch(color=COL["blue"], label="of which: some rotation put max p > 0.9 on its answer")],
              loc="upper left", bbox_to_anchor=(0, -0.3), frameon=False, fontsize=8, ncol=2)
    ax.set_title("Order-flip rate by question and model, 3 cyclic rotations", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "order-sensitivity-flip-rate.png")
    plt.close(fig)


def fig_averaging(summary, figdir):
    plt = _plt()
    from matplotlib.lines import Line2D
    panels = _panels(summary)
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    ys = np.arange(len(panels))[::-1]
    for yy, (label, r) in zip(ys, panels):
        singles = [100 * a for a in r["acc_single_rotation"]]
        ax.plot([min(singles), max(singles)], [yy, yy], color=COL["range"], linewidth=7, solid_capstyle="round", zorder=1)
        ax.scatter(singles, [yy] * len(singles), s=44, color=COL["gray"], edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter([100 * r["acc_oracle_any"]], [yy], s=70, marker="o", facecolor="white", edgecolor=COL["gray"], linewidth=1.4, zorder=2)
        ax.scatter([100 * r["acc_majority"]], [yy], s=70, marker="s", facecolor="white", edgecolor=COL["blue"], linewidth=1.6, zorder=4)
        ax.scatter([100 * r["acc_averaged"]], [yy], s=95, marker="D", color=COL["blue"], edgecolor="white", linewidth=1.5, zorder=5)
        ax.text(100 * r["acc_averaged"], yy + 0.28, f"averaged {100 * r['acc_averaged']:.1f}", ha="center", fontsize=8, color=COL["ink"])
        ax.text(100 * r["acc_oracle_any"] + 1.2, yy, f"any rotation right: {100 * r['acc_oracle_any']:.0f}", va="center", fontsize=7.5, color=COL["muted"])
        ax.text(min(singles) - 1.2, yy, f"worst rotation {min(singles):.0f}", va="center", ha="right", fontsize=7.5, color=COL["muted"])
    ax.set_yticks(ys)
    ax.set_yticklabels([p[0] for p in panels], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(panels) - 0.3)
    ax.set_xlabel("accuracy, %")
    ax.xaxis.grid(True, color=COL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", color=COL["gray"], linestyle="none", markersize=6, label="one rotation alone (three dots)"),
                       Line2D([], [], marker="D", color=COL["blue"], linestyle="none", markersize=7, label="3 rotations averaged (the study's number)"),
                       Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=COL["blue"], color="none", linestyle="none", markersize=7, label="plurality vote of the 3 argmaxes"),
                       Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=COL["gray"], color="none", linestyle="none", markersize=7, label="any rotation right (upper bound, not a method)")],
              loc="upper left", bbox_to_anchor=(0, -0.22), frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("What averaging buys: each rotation alone vs the 3-rotation average", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "order-sensitivity-averaging.png")
    plt.close(fig)


def fig_position_bias(summary, figdir):
    plt = _plt()
    panels = _panels(summary)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4), dpi=150, gridspec_kw={"width_ratios": [8, 24]})
    for ax, (label, r) in zip(axes.ravel(), panels):
        ps = r["position_shares"]
        k = r["k"]
        x = np.arange(k)
        win = 100 * np.array(ps["win"])
        gold = 100 * np.array(ps["gold"])
        ax.bar(x, win, width=0.72, color=COL["blue"], edgecolor="white", linewidth=1, label="share of wins at this position")
        ax.scatter(x, gold, s=26, color=COL["gold"], edgecolor="white", linewidth=1, zorder=3, label="share of gold labels at this position")
        ax.set_xticks(x)
        ax.set_xticklabels(list(ps["labels"]), fontsize=7 if k > 8 else 8)
        ax.set_title(label, loc="left", fontsize=8.5)
        ax.yaxis.grid(True, color=COL["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        top = max(win.max(), gold.max())
        ax.set_ylim(0, top * 1.3)
        j = int(win.argmax())
        ha = "left" if j < k / 3 else ("right" if j > 2 * k / 3 else "center")
        xt = j - 0.4 if ha == "left" else (j + 0.4 if ha == "right" else j)
        ax.text(xt, win[j] + top * 0.05, f"{ps['labels'][j]}: {win[j]:.0f}% of wins, {gold[j]:.0f}% of gold", ha=ha, fontsize=7.5, color=COL["ink"])
    axes[0, 0].set_ylabel("share, %")
    axes[1, 0].set_ylabel("share, %")
    axes[1, 0].set_xlabel("display position (letter)")
    axes[1, 1].set_xlabel("display position (letter)")
    axes[0, 1].legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.suptitle("Where the wins land: display position of the argmax vs of the gold, pooled over rotations × rows", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(figdir / "order-sensitivity-position-bias.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/order-sensitivity.json")
    ap.add_argument("--md", default="results/summary/order-sensitivity.md")
    ap.add_argument("--figdir", default="results/figures")
    a = ap.parse_args()
    summary = {"generated": dt.date.today().isoformat(), "definitions": DEFINITIONS,
               "files": [analyze_file(p, m, l) for p, m, l in FILES if os.path.exists(p)]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    md = to_markdown(summary)
    open(a.md, "w").write(md)
    figdir = Path(a.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    fig_flip_rate(summary, figdir)
    fig_averaging(summary, figdir)
    fig_position_bias(summary, figdir)
    print(md)
    print("wrote", a.out, a.md, "and figures under", figdir)
