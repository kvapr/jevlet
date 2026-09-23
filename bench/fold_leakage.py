"""Fold leakage check for out-of-fold temperature scaling.

The reported calibration numbers fit one temperature on four of five folds GROUPED by seed request
(jevlet.calibrate.cv_temperature, folds from group_folds) and apply it to the fifth. This script
asks what changes when the folds are plain random row folds instead, so that paraphrases of one
seed sit in the fitting folds while their sibling is being scored.

Per result file and question it computes:
  (a) grouped 5-fold OOF temperature scaling: ECE / NLL / Brier, per-fold temperatures
  (b) random-row 5-fold OOF temperature scaling: the same
  (c) the leak: fraction of held-out rows whose base_input has siblings in the fitting folds
      under random splitting (zero by construction under grouped splitting)
  (d) per-fold temperatures both ways (seed 0 = the folds bench.analyze uses)
  (e) (a) and (b) repeated over N seeds of the fold assignment: mean, sd, min, max
  plus the in-sample reference (one temperature fitted on all rows, applied to all rows: the
  fully leaked extreme), the selective numbers at Jev confidence 0.9, and a within-group
  agreement measure that says how near-duplicate the paraphrases are in the model's eyes.

    python -m bench.fold_leakage results/router-*.json --seeds 20 \
        --out results/fold-leakage.json --md results/summary/fold-leakage.md --fig results/figures/fold-leakage

Nothing here touches the tracked result files; it only reads them.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

from jevlet import metrics as M
from jevlet.calibrate import cv_temperature, fit_temperature, fit_vector_scaling, group_folds, softmax
from bench.analyze import _labels_for

N_FOLDS = 5


def _load(path):
    d = json.load(open(path))
    meta, res = d["meta"], d["results"]
    rows = meta["rows"]
    groups_all = [r["base_input"] for r in rows]
    out = {}
    for qid, q in meta["questions"].items():
        recs = sorted(res[qid], key=lambda x: x["i"])
        P = np.array([x["probs"] for x in recs], dtype=float)
        L = np.array([x["logits"] if x.get("logits") else np.log(np.clip(x["probs"], 1e-9, 1)) for x in recs], dtype=float)
        y = _labels_for(qid, q, rows)
        valid = y >= 0
        P, L, y = P[valid], L[valid], y[valid]
        g = [groups_all[i] for i in np.where(valid)[0]]
        out[qid] = (P, L, y, g)
    return meta.get("backend"), out


def _oof(L, y, folds):
    """Out-of-fold temperature scaling for a given fold vector. Returns (probs, temps)."""
    out = np.zeros_like(L)
    temps = []
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        if te.sum() == 0:
            continue
        T = fit_temperature(L[tr], y[tr]) if tr.sum() else 1.0
        temps.append(T)
        out[te] = softmax(L[te] / T)
    return out, temps


def _oof_vector(L, y, folds):
    out = np.zeros_like(L)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        if te.sum() == 0:
            continue
        w, b = fit_vector_scaling(L[tr], y[tr])
        out[te] = softmax(L[te] * w + b)
    return out


def _leak(groups, folds):
    """Under these folds: fraction of rows with at least one same-group row in a different fold,
    and the mean number of such siblings per row."""
    by_g = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by_g[g].append(i)
    leaked, sib = 0, 0
    for i, g in enumerate(groups):
        others = [j for j in by_g[g] if j != i and folds[j] != folds[i]]
        leaked += bool(others)
        sib += len(others)
    n = len(groups)
    return leaked / n, sib / n


def _metrics(P, y):
    s = M.summarize(P, y)
    sel = {str(x["tau"]): {"n": x["n"], "wrong": x["wrong"], "coverage": x["coverage"]} for x in s["selective_jev"]}
    return {"ece": s["ece"], "ece_adaptive": s["ece_adaptive"], "nll": s["nll"], "brier": s["brier"],
            "accuracy": s["accuracy"], "mean_conf": s["mean_conf"], "selective_jev": sel}


def _agg(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def _within_group(P, y, groups):
    """How alike are the paraphrases of one seed in the model's eyes? Over groups with >= 2 rows:
    share of groups whose rows all get the same predicted answer, share whose rows are all right or
    all wrong, and the within-group sd of raw confidence relative to the overall sd."""
    by_g = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by_g[g].append(i)
    multi = [idx for idx in by_g.values() if len(idx) >= 2]
    if not multi:
        return {"n_multi_groups": 0}
    pred, conf, corr = P.argmax(1), P.max(1), P.argmax(1) == y
    same_pred = np.mean([len(set(pred[idx])) == 1 for idx in multi])
    same_corr = np.mean([len(set(corr[idx])) == 1 for idx in multi])
    wsd = np.mean([conf[idx].std() for idx in multi])
    return {"n_multi_groups": len(multi), "rows_in_multi_groups": int(sum(len(i) for i in multi)),
            "share_same_prediction": float(same_pred), "share_same_correctness": float(same_corr),
            "within_group_conf_sd": float(wsd), "overall_conf_sd": float(conf.std())}


def run_file(path, seeds):
    backend, qs = _load(path)
    rows_n = None
    out = {"file": os.path.basename(path), "backend": backend, "n_folds": N_FOLDS, "seeds": seeds, "questions": {}}
    for qid, (P, L, y, g) in qs.items():
        n = len(y)
        rows_n = n
        gsize = collections.Counter(collections.Counter(g).values())
        r = {"n": n, "n_groups": len(set(g)), "group_size_hist": {str(k): v for k, v in sorted(gsize.items())},
             "rows_in_multi_row_groups": int(sum(k * v for k, v in gsize.items() if k > 1)),
             "raw": _metrics(P, y), "within_group": _within_group(P, y, g)}
        # in-sample reference: one T on all rows, applied to all rows
        T_all = fit_temperature(L, y)
        r["in_sample"] = dict(_metrics(softmax(L / T_all), y), temperature=T_all)
        # grouped and random, over seeds
        per = {"grouped": [], "random": []}
        for s in range(seeds):
            fg = group_folds(g, N_FOLDS, seed=s)
            fr = group_folds(list(range(n)), N_FOLDS, seed=s)
            for name, folds in (("grouped", fg), ("random", fr)):
                Pc, temps = _oof(L, y, folds)
                leak_frac, leak_sib = _leak(g, folds)
                rec = dict(_metrics(Pc, y), seed=s, temperatures=temps, leak_fraction=leak_frac,
                           leak_siblings_per_row=leak_sib,
                           fold_sizes=[int((folds == f).sum()) for f in range(N_FOLDS)])
                if n >= 8 * L.shape[1]:   # vector scaling only where calibrate.py would not fall back
                    Pv = _oof_vector(L, y, folds)
                    rec["vector_ece"] = M.ece(Pv.max(1), Pv.argmax(1) == y)
                    rec["vector_nll"] = M.nll(Pv, y)
                per[name].append(rec)
        for name in ("grouped", "random"):
            recs = per[name]
            agg = {k: _agg([x[k] for x in recs]) for k in ("ece", "ece_adaptive", "nll", "brier", "leak_fraction")}
            agg["wrong_at_0.9"] = _agg([x["selective_jev"]["0.9"]["wrong"] for x in recs])
            agg["coverage_at_0.9"] = _agg([x["selective_jev"]["0.9"]["coverage"] for x in recs])
            allT = [t for x in recs for t in x["temperatures"]]
            agg["temperature"] = _agg(allT)
            agg["temperature_fold_range_mean"] = float(np.mean([max(x["temperatures"]) - min(x["temperatures"]) for x in recs]))
            agg["temperature_at_grid_floor"] = int(sum(t <= 0.0501 for t in allT))
            if "vector_ece" in recs[0]:
                agg["vector_ece"] = _agg([x["vector_ece"] for x in recs])
            r[name] = {"seed0": recs[0], "per_seed": recs, "over_seeds": agg}
        # paired difference grouped - random, same seed
        d_ece = [a["ece"] - b["ece"] for a, b in zip(per["grouped"], per["random"])]
        d_nll = [a["nll"] - b["nll"] for a, b in zip(per["grouped"], per["random"])]
        r["paired_grouped_minus_random"] = {"ece": _agg(d_ece), "nll": _agg(d_nll),
                                            "seeds_grouped_ece_higher": int(sum(x > 0 for x in d_ece))}
        out["questions"][qid] = r
    out["n"] = rows_n
    return out


def to_markdown(results):
    lines = ["# Fold leakage: grouped vs random folds for out-of-fold temperature scaling", "",
             "Seed 0 of the grouped folds is the fold assignment bench.analyze uses. `leak` = share of rows whose seed",
             "request has paraphrases in the fitting folds. Spread = mean ± sd over the seeds of the fold assignment.", "",
             "| file | question | n | groups | leak (random) | ECE raw | ECE grouped s0 | ECE random s0 | ECE grouped (spread) | ECE random (spread) | ECE in-sample | NLL grouped / random | T grouped s0 | T random s0 | wrong@0.9 raw → grouped / random |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for res in results:
        for qid, r in res["questions"].items():
            g, rd = r["grouped"], r["random"]
            lines.append(
                f"| {res['file']} | {qid} | {r['n']} | {r['n_groups']} | {rd['over_seeds']['leak_fraction']['mean']:.3f} | "
                f"{r['raw']['ece']:.4f} | {g['seed0']['ece']:.4f} | {rd['seed0']['ece']:.4f} | "
                f"{g['over_seeds']['ece']['mean']:.4f} ± {g['over_seeds']['ece']['sd']:.4f} | "
                f"{rd['over_seeds']['ece']['mean']:.4f} ± {rd['over_seeds']['ece']['sd']:.4f} | {r['in_sample']['ece']:.4f} | "
                f"{g['over_seeds']['nll']['mean']:.3f} / {rd['over_seeds']['nll']['mean']:.3f} | "
                f"{[round(t, 2) for t in g['seed0']['temperatures']]} | {[round(t, 2) for t in rd['seed0']['temperatures']]} | "
                f"{r['raw']['selective_jev']['0.9']['wrong']} → {g['seed0']['selective_jev']['0.9']['wrong']} / {rd['seed0']['selective_jev']['0.9']['wrong']} |")
    return "\n".join(lines)


def figures(results, prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    C_G, C_R, C_I = "#2a6db3", "#c7831c", "#6b7378"
    short = {"router-qwen3-4b-q4_0-cpu-rot3.json": "readout 4B\n260 base rows",
             "router-qwen3-4b-q4_0-cpu-all-rot3.json": "readout 4B\nall 519 rows",
             "router-qwen3-4b-q4_0-cpu-authored-rot3.json": "readout 4B\n259 authored rows",
             "router-ft-modernbert-base-desc-shuf-e8.json": "fine-tuned 150M\n126 held-out rows",
             "router-laya-td-short-all.json": "Laya 0-shot\n519 rows"}
    qids = ["tool", "command", "is_write", "touches_poe_power"]
    # ---- figure 1: ECE grouped vs random, one panel per file
    fig, axes = plt.subplots(1, len(results), figsize=(8, 3.6), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results):
        for k, qid in enumerate(qids):
            r = res["questions"][qid]
            g, rd = r["grouped"]["over_seeds"]["ece"], r["random"]["over_seeds"]["ece"]
            ax.plot([g["min"], g["max"]], [k + 0.16, k + 0.16], color=C_G, lw=2, solid_capstyle="round")
            ax.plot([rd["min"], rd["max"]], [k - 0.16, k - 0.16], color=C_R, lw=2, solid_capstyle="round")
            ax.plot(g["mean"], k + 0.16, "o", color=C_G, ms=5, mec="white", mew=1)
            ax.plot(rd["mean"], k - 0.16, "o", color=C_R, ms=5, mec="white", mew=1)
            ax.plot(r["in_sample"]["ece"], k, "|", color=C_I, ms=9, mew=1.4)
        ax.set_yticks(range(len(qids)))
        ax.set_yticklabels(["tool", "command", "is_write", "touches_poe"], fontsize=8)
        ax.set_title(short.get(res["file"], res["file"]), fontsize=8)
        ax.set_xlim(0, 0.16)
        ax.set_xticks([0, 0.05, 0.10, 0.15])
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", color="#e4e8e6", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#c9d1ce"); ax.spines["bottom"].set_color("#c9d1ce")
        ax.invert_yaxis()
    axes[0].set_ylabel("question", fontsize=8)
    fig.supxlabel("ECE after out-of-fold temperature scaling (10 equal-width bins); dot = mean, bar = min to max over 20 fold seeds", fontsize=7.5, y=0.02)
    fig.legend(handles=[plt.Line2D([], [], color=C_G, marker="o", lw=2, ms=5, label="folds grouped by seed request"),
                        plt.Line2D([], [], color=C_R, marker="o", lw=2, ms=5, label="random row folds (paraphrases leak)"),
                        plt.Line2D([], [], color=C_I, marker="|", lw=0, ms=9, mew=1.4, label="in-sample (fit on all rows)")],
               loc="upper center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(f"{prefix}-ece.png", facecolor="white")
    plt.close(fig)
    # ---- figure 2: per-fold temperatures, all seeds, grouped vs random
    fig, axes = plt.subplots(1, len(results), figsize=(8, 3.6), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, res in zip(axes, results):
        for k, qid in enumerate(qids):
            r = res["questions"][qid]
            tg = [t for x in r["grouped"]["per_seed"] for t in x["temperatures"]]
            tr = [t for x in r["random"]["per_seed"] for t in x["temperatures"]]
            ax.scatter(tg, k + 0.16 + rng.uniform(-0.07, 0.07, len(tg)), s=7, color=C_G, alpha=0.45, lw=0)
            ax.scatter(tr, k - 0.16 + rng.uniform(-0.07, 0.07, len(tr)), s=7, color=C_R, alpha=0.45, lw=0)
            ax.plot([r["in_sample"]["temperature"]] * 2, [k - 0.3, k + 0.3], color=C_I, lw=1)
        ax.set_xscale("log")
        ax.set_xlim(0.04, 60)
        ax.set_xticks([0.05, 0.2, 1, 5, 20])
        ax.set_xticklabels(["0.05", "0.2", "1", "5", "20"], fontsize=7)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_yticks(range(len(qids)))
        ax.set_yticklabels(["tool", "command", "is_write", "touches_poe"], fontsize=8)
        ax.set_title(short.get(res["file"], res["file"]), fontsize=8)
        ax.grid(axis="x", color="#e4e8e6", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#c9d1ce"); ax.spines["bottom"].set_color("#c9d1ce")
        ax.invert_yaxis()
    axes[0].set_ylabel("question", fontsize=8)
    fig.supxlabel("fitted temperature per fold (log scale); 100 fits per question and method = 5 folds × 20 seeds; grey bar = in-sample fit", fontsize=7.5, y=0.02)
    fig.legend(handles=[plt.Line2D([], [], color=C_G, marker="o", lw=0, ms=5, label="folds grouped by seed request"),
                        plt.Line2D([], [], color=C_R, marker="o", lw=0, ms=5, label="random row folds"),
                        plt.Line2D([], [], color=C_I, lw=1, label="in-sample")],
               loc="upper center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(f"{prefix}-temps.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/fold-leakage.json")
    ap.add_argument("--md", default="results/summary/fold-leakage.md")
    ap.add_argument("--fig", default="results/figures/fold-leakage", help="figure path prefix")
    a = ap.parse_args()
    results = [run_file(f, a.seeds) for f in a.files]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_folds": N_FOLDS, "seeds": a.seeds, "files": results}, open(a.out, "w"), indent=1)
    md = to_markdown(results)
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    open(a.md, "w").write(md + "\n")
    figures(results, a.fig)
    print(md)
