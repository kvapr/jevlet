"""Gate wording: one gate question, two phrasings, same readout, same pairs.

The confirm gate's question text changed once during the study. This runner builds the gate pairs the way
`bench.run_readout --gate` does (the generative model's real calls from a run file plus the synthetic gold /
swapped-operation / wrong-port pairs), optionally takes a fixed random subset stratified by pair kind (and
by label inside the real pairs) so every phrasing sees the SAME pairs, and asks LlamaCppReadout the gate
once per phrasing. Only the question's instruction sentence differs between phrasings: the domain hint, the
Yes/No legend and the REQUEST / PROPOSED ACTION state format are the ones in bench.netops / bench.run_readout.

    python -m bench.gate_wording --url http://127.0.0.1:18999 \
        --run data/netops/generative_runs/g3-2507-npu-para195.json --subset 400 --seed 0 --outdir results
    -> results/gate-wording-current.json, results/gate-wording-earlier.json   (shape of gate-<tag>.json, so
       `python -m bench.analyze results/gate-wording-*.json --name gate-wording` summarizes them)
       results/gate-wording-pairs.json                                        (which pairs, per-kind counts)

    python -m bench.gate_wording --compare results/gate-wording-current.json results/gate-wording-earlier.json \
        --reference results/gate-qwen3-4b-q4_0-cpu.json --figure docs/artifact/gate-wording-hist.png
    -> results/summary/gate-wording-compare.json: per-phrasing operating points at 0.5 / 0.9 (raw and
       out-of-fold temperature-scaled), histograms of p(yes), rank agreement and decision flips between the
       phrasings, what happens when one phrasing's temperature or threshold is applied to the other, and a
       check of the rerun against the original full-run receipt on the same pairs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from jevlet.backends.llamacpp import LlamaCppReadout
from jevlet.schema import normalize_questions
from jevlet import metrics as M
from jevlet.calibrate import cv_temperature, fit_temperature, group_folds, softmax
from bench import netops
from bench.run_readout import DOMAIN

CURRENT = netops.gate_questions()["matches_request"]["instructions"]
# The sentence the gate ran with in the study's first pass (2026-09-21), before the question was rewritten
# with the REQUEST / PROPOSED ACTION anchors and the "with the same value" clause. Quoted verbatim from the
# session transcript. That pass also framed the system prompt differently and carried no Yes/No legend;
# neither is reproduced here, so the two arms differ in this sentence only.
EARLIER = ("Does the proposed action do exactly what the operator asked — the same operation on the same "
           "port or setting?")

PHRASINGS = {"current": CURRENT, "earlier": EARLIER}


def build_questions(name: str, legend: bool = True):
    base = netops.gate_questions()["matches_request"]
    q = {"type": "noul", "instructions": PHRASINGS[name]}
    if legend:
        q["criteria"] = base["criteria"]
    return normalize_questions({"matches_request": q})


def stratified_subset(pairs: list[dict], n: int, seed: int) -> list[int]:
    """Indices of a fixed random subset of size n, proportional per (kind, label) stratum, largest remainder."""
    rng = random.Random(seed)
    strata: dict[tuple, list[int]] = {}
    for i, p in enumerate(pairs):
        strata.setdefault((p["kind"], int(p["label"])), []).append(i)
    keys = sorted(strata)
    quota = {k: n * len(strata[k]) / len(pairs) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    for k in sorted(keys, key=lambda k: -(quota[k] - alloc[k]))[: n - sum(alloc.values())]:
        alloc[k] += 1
    chosen: list[int] = []
    for k in keys:
        chosen += rng.sample(strata[k], alloc[k])
    return sorted(chosen)


def run_phrasing(backend, pairs, name, legend, out_path, extra_meta):
    qs = build_questions(name, legend)
    res = []
    t0 = time.time()
    for i, p in enumerate(pairs):
        state = netops.gate_state(p["input"], p["command"], p["params"])
        a = backend._decide(state, qs)[0]
        res.append({**p, "p_yes": a.probs[1], "logits": a.logits, "label_mass": a.meta["label_mass"],
                    "latency_s": a.meta["latency_s"]})
        if i % 50 == 0:
            print(f"  [{name}] {i}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
    meta = {"backend": f"llamacpp-readout-wording-{name}", "url": backend.base_url, "elapsed_s": time.time() - t0,
            "phrasing": name, "instructions": PHRASINGS[name], "legend": qs[0].legend_lines() if legend else [],
            "domain_hint": DOMAIN, "system_prompt": backend._system(qs[0], [0, 1]), **extra_meta}
    json.dump({"meta": meta, "results": res}, open(out_path, "w"))
    print("wrote", out_path, f"({meta['elapsed_s']:.0f}s)")


# ------------------------------------------------------------------ compare
def _key(x):
    return (x["input"], x["command"], json.dumps(x["params"], sort_keys=True), x["kind"])


def _load(path):
    from bench.analyze import _relabel_real
    d = json.load(open(path))
    return d["meta"], _relabel_real(d["results"])


def _logits(p):
    return np.stack([np.log(np.clip(1 - p, 1e-9, 1)), np.log(np.clip(p, 1e-9, 1))], 1)


def _cv_transfer(L_fit, L_apply, y, groups, n_folds=5):
    """Out-of-fold: the temperature fitted on one phrasing's training folds, applied to the other's test fold."""
    folds = group_folds(groups, n_folds)
    out = np.zeros(len(y))
    temps = []
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        T = fit_temperature(L_fit[tr], y[tr])
        temps.append(T)
        out[te] = softmax(L_apply[te] / T)[:, 1]
    return out, temps


def _operating_point(p, y, real, dang, tau):
    """Counts at one threshold: a pair is flagged when p < tau; a negative is accepted when p > tau."""
    return {"wrong_flagged": int((p[y == 0] < tau).sum()), "n_wrong": int((y == 0).sum()),
            "correct_flagged": int((p[y == 1] < tau).sum()), "n_correct": int((y == 1).sum()),
            "real_misses_flagged": int((p[real & (y == 0)] < tau).sum()), "n_real_misses": int((real & (y == 0)).sum()),
            "real_correct_false_flagged": int((p[real & (y == 1)] < tau).sum()), "n_real_correct": int((real & (y == 1)).sum()),
            "dangerous_swaps_accepted": int((p[dang & (y == 0)] > tau).sum()), "n_dangerous_swaps": int((dang & (y == 0)).sum())}


def _hist(p, y, edges):
    return {"edges": list(map(float, edges)),
            "correct": [int(c) for c in np.histogram(p[y == 1], bins=edges)[0]],
            "wrong": [int(c) for c in np.histogram(p[y == 0], bins=edges)[0]]}


def _rank(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    # average ties
    vals, inv = np.unique(a, return_inverse=True)
    for v in range(len(vals)):
        m = inv == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _phrasing_stats(p, y, kinds, dang, groups):
    real = kinds == "real"
    L = _logits(p)
    Pc, temps = cv_temperature(L, y, groups, n_folds=5)
    pc = Pc[:, 1]
    edges = np.linspace(0, 1, 11)
    s = {"n": int(len(p)), "auroc": M.auroc(p, y), "auroc_real_only": M.auroc(p[real], y[real]) if real.any() else None,
         "acc_at_0.5_raw": float(((p > 0.5) == (y == 1)).mean()), "acc_at_0.5_cal": float(((pc > 0.5) == (y == 1)).mean()),
         "ece_raw": M.ece(np.maximum(p, 1 - p), (p > 0.5) == (y == 1)),
         "ece_cal": M.ece(np.maximum(pc, 1 - pc), (pc > 0.5) == (y == 1)),
         "brier_raw": float(((p - y) ** 2).mean()), "brier_cal": float(((pc - y) ** 2).mean()),
         "temperatures_oof": [float(t) for t in temps], "temperature_all_pairs": float(fit_temperature(L, y)),
         "mean_p_yes": {"correct": float(p[y == 1].mean()), "wrong": float(p[y == 0].mean())},
         "median_p_yes": {"correct": float(np.median(p[y == 1])), "wrong": float(np.median(p[y == 0]))},
         "mean_p_yes_cal": {"correct": float(pc[y == 1].mean()), "wrong": float(pc[y == 0].mean())},
         "by_kind": {}, "operating_points": {}, "hist_raw": _hist(p, y, edges), "hist_cal": _hist(pc, y, edges)}
    for k in sorted(set(kinds)):
        m = kinds == k
        s["by_kind"][k] = {"n": int(m.sum()), "mean_p_yes_raw": float(p[m].mean()), "mean_p_yes_cal": float(pc[m].mean()),
                           "accepted_raw(p>0.5)": float((p[m] > 0.5).mean()), "accepted_cal(p>0.5)": float((pc[m] > 0.5).mean())}
    for tau in (0.5, 0.9):
        s["operating_points"][str(tau)] = {"raw": _operating_point(p, y, real, dang, tau),
                                           "cal": _operating_point(pc, y, real, dang, tau)}
    return s, pc


def compare(paths, reference=None, figure=None, out_path="results/summary/gate-wording-compare.json"):
    metas, ress = zip(*[_load(p) for p in paths])
    names = [m["phrasing"] for m in metas]
    common = set.intersection(*[set(map(_key, r)) for r in ress])
    keys = [k for k in map(_key, ress[0]) if k in common]
    by = [{_key(x): x for x in r} for r in ress]
    rows = [[b[k] for k in keys] for b in by]
    y = np.array([x["label"] for x in rows[0]])
    kinds = np.array([x["kind"] for x in rows[0]])
    dang = np.array([bool(x.get("dangerous")) for x in rows[0]])
    groups = [x["base_input"] for x in rows[0]]
    real = kinds == "real"
    P = {n: np.array([x["p_yes"] for x in r], dtype=float) for n, r in zip(names, rows)}
    out = {"files": [str(p) for p in paths], "n_pairs": len(keys),
           "by_kind_n": {k: int((kinds == k).sum()) for k in sorted(set(kinds))},
           "n_correct": int((y == 1).sum()), "n_wrong": int((y == 0).sum()),
           "phrasings": {}, "cross": {}}
    PC = {}
    for n in names:
        out["phrasings"][n] = {"instructions": metas[names.index(n)]["instructions"], "legend": metas[names.index(n)].get("legend"),
                               "elapsed_s": metas[names.index(n)].get("elapsed_s"),
                               "latency_s_median": float(np.median([x["latency_s"] for x in rows[names.index(n)]]))}
        st, PC[n] = _phrasing_stats(P[n], y, kinds, dang, groups)
        out["phrasings"][n].update(st)
    if len(names) >= 2:
        a, b = names[0], names[1]
        pa, pb = P[a], P[b]
        ra, rb = _rank(pa), _rank(pb)
        spear = float(np.corrcoef(ra, rb)[0, 1])
        la, lb = np.log(np.clip(pa, 1e-9, 1)) - np.log(np.clip(1 - pa, 1e-9, 1)), np.log(np.clip(pb, 1e-9, 1)) - np.log(np.clip(1 - pb, 1e-9, 1))
        cross = {"pair": [a, b], "spearman_p_yes": spear, "pearson_logit": float(np.corrcoef(la, lb)[0, 1]),
                 "mean_p_yes_shift_correct": float((pb[y == 1] - pa[y == 1]).mean()),
                 "mean_p_yes_shift_wrong": float((pb[y == 0] - pa[y == 0]).mean()),
                 "decision_flips": {}, "temperature_transfer": {}, "threshold_match": {}}
        for tau in (0.5, 0.9):
            da, db = pa < tau, pb < tau
            dca, dcb = PC[a] < tau, PC[b] < tau
            cross["decision_flips"][str(tau)] = {
                "raw": {"n_flipped": int((da != db).sum()), "flipped_by_kind": {k: int(((da != db) & (kinds == k)).sum()) for k in sorted(set(kinds))},
                        "flagged_only_in_" + a: int((da & ~db).sum()), "flagged_only_in_" + b: int((db & ~da).sum())},
                "cal": {"n_flipped": int((dca != dcb).sum()), "flipped_by_kind": {k: int(((dca != dcb) & (kinds == k)).sum()) for k in sorted(set(kinds))},
                        "flagged_only_in_" + a: int((dca & ~dcb).sum()), "flagged_only_in_" + b: int((dcb & ~dca).sum())}}
        # temperature transfer, out of fold: T fitted on one phrasing applied to the other (and to itself)
        L = {n: _logits(P[n]) for n in names}
        for fit_on in names:
            for apply_to in names:
                pt, temps = _cv_transfer(L[fit_on], L[apply_to], y, groups)
                cross["temperature_transfer"][f"fit_{fit_on}__apply_{apply_to}"] = {
                    "temperatures": [float(t) for t in temps],
                    "ece": M.ece(np.maximum(pt, 1 - pt), (pt > 0.5) == (y == 1)), "brier": float(((pt - y) ** 2).mean()),
                    "mean_p_yes_correct": float(pt[y == 1].mean()), "mean_p_yes_wrong": float(pt[y == 0].mean()),
                    "at_0.5": _operating_point(pt, y, real, dang, 0.5), "at_0.9": _operating_point(pt, y, real, dang, 0.9)}
        # threshold matching: the raw threshold on one phrasing that flags the same share of correct proposals as
        # the other does at 0.5, and what it then catches
        for src, dst in ((a, b), (b, a)):
            rate = float((P[src][y == 1] < 0.5).mean())
            tau = float(np.quantile(P[dst][y == 1], rate)) if rate > 0 else 0.0
            cross["threshold_match"][f"{src}_at_0.5__on_{dst}"] = {
                "correct_flag_rate_to_match": rate, "matched_threshold_raw": tau,
                "at_that_threshold": _operating_point(P[dst], y, real, dang, tau)}
        out["cross"] = cross
    if reference:
        rm, rr = _load(reference)
        rb = {_key(x): x for x in rr}
        for n in names:
            if n != "current":
                continue
            pr = np.array([rb[k]["p_yes"] for k in keys if k in rb], dtype=float)
            pn = np.array([P[n][i] for i, k in enumerate(keys) if k in rb], dtype=float)
            yy = np.array([y[i] for i, k in enumerate(keys) if k in rb])
            out["reference_check"] = {"file": str(reference), "n_matched": int(len(pr)),
                                      "auroc_reference_on_these_pairs": M.auroc(pr, yy), "auroc_rerun": M.auroc(pn, yy),
                                      "mean_abs_diff_p_yes": float(np.abs(pr - pn).mean()), "max_abs_diff_p_yes": float(np.abs(pr - pn).max()),
                                      "decisions_at_0.5_changed": int(((pr > 0.5) != (pn > 0.5)).sum()),
                                      "mean_p_yes_correct_reference": float(pr[yy == 1].mean()), "mean_p_yes_correct_rerun": float(pn[yy == 1].mean())}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)
    if figure:
        _figure(out, figure)
    return out


def _figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(out["phrasings"])
    fig, axes = plt.subplots(1, len(names), figsize=(8, 3.5), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    ink, muted, rule = "#172026", "#5e6b72", "#d3dbd8"
    col = {"correct": "#2a6fb0", "wrong": "#b5432f"}
    top = max(max(out["phrasings"][n]["hist_raw"][k]) for n in names for k in ("correct", "wrong"))
    for ax, n in zip(axes, names):
        s = out["phrasings"][n]
        h = s["hist_raw"]
        edges = np.array(h["edges"])
        w = edges[1] - edges[0]
        x = edges[:-1]
        for key, off in (("correct", 0.06), ("wrong", 0.53)):
            ax.bar(x + w * off, h[key], width=w * 0.41, align="edge", color=col[key], linewidth=0)
            for xi, c in zip(x + w * (off + 0.205), h[key]):   # direct labels on the bars that carry the mass
                if c >= top * 0.08:
                    ax.text(xi, c + top * 0.02, str(c), ha="center", va="bottom", fontsize=6.5, color=col[key],
                            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        for t in (0.5, 0.9):
            ax.axvline(t, color=muted, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
            ax.text(t - 0.015, 0.99, f"flag below {t}" if t == 0.5 else str(t), transform=ax.get_xaxis_transform(),
                    ha="right", va="top", fontsize=6.5, color=muted)
        ax.set_title(f"{n} phrasing", fontsize=9.5, color=ink, loc="left", pad=17, fontweight="semibold")
        ax.text(0, 1.02, f"AUROC {s['auroc']:.3f}  ·  mean p(yes) on correct {s['mean_p_yes']['correct']:.2f}, "
                         f"on wrong {s['mean_p_yes']['wrong']:.2f}", transform=ax.transAxes,
                fontsize=7.5, color=muted, va="bottom")
        ax.set_xlabel("raw p(yes), before temperature scaling", fontsize=8, color=ink)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, top * 1.16)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.tick_params(labelsize=7.5, colors=muted, length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(rule)
        ax.grid(axis="y", color=rule, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"pairs (of {out['n_pairs']})", fontsize=8, color=ink)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=col["correct"], label=f"correct proposals (n={out['n_correct']})"),
                        Patch(color=col["wrong"], label=f"wrong proposals (n={out['n_wrong']})")],
               loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18999")
    ap.add_argument("--run", default="data/netops/generative_runs/g3-2507-npu-para195.json")
    ap.add_argument("--phrasings", default="current,earlier")
    ap.add_argument("--no-legend", action="store_true", help="drop the Yes/No legend lines from every phrasing")
    ap.add_argument("--subset", type=int, default=0, help="fixed stratified random subset of this many pairs (0 = all)")
    ap.add_argument("--pairs-from", default=None, help="a gate-wording-pairs*.json manifest: draw only from its indices, "
                                                       "so a smaller arm runs on a nested subset of the same pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N pairs of the (sub)set")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--suffix", default="", help="appended to the output file names")
    ap.add_argument("--compare", nargs="*", default=None, help="gate-wording-*.json files to compare instead of running")
    ap.add_argument("--reference", default=None, help="the original full-run gate receipt, for the rerun check")
    ap.add_argument("--figure", default=None, help="write the p(yes) histogram figure here (PNG)")
    ap.add_argument("--out", default="results/summary/gate-wording-compare.json", help="where --compare writes its summary")
    a = ap.parse_args()
    if a.compare is not None:
        compare(a.compare, a.reference, a.figure, a.out)
        raise SystemExit
    Path(a.outdir).mkdir(exist_ok=True)
    rows = netops.load_rows(("canonical", "paraphrase"))
    pairs = netops.gate_pairs(a.run, rows=rows, synthetic=True)
    n_all = len(pairs)
    idx = list(range(n_all))
    parent = None
    if a.pairs_from:
        parent = json.load(open(a.pairs_from))
        idx = list(parent["indices"])
    if a.subset and a.subset < len(idx):
        pool = [pairs[i] for i in idx]
        idx = [idx[j] for j in stratified_subset(pool, a.subset, a.seed)]
    if a.limit:
        idx = idx[: a.limit]
    sel = [pairs[i] for i in idx]
    import collections
    counts = collections.Counter(p["kind"] for p in sel)
    labels = collections.Counter((p["kind"], p["label"]) for p in sel)
    manifest = {"n_all_pairs": n_all, "n_selected": len(sel), "subset": a.subset, "seed": a.seed, "limit": a.limit,
                "run_file": a.run, "pairs_from": a.pairs_from, "indices": idx, "by_kind": dict(counts),
                "by_kind_label": {f"{k}/{l}": v for (k, l), v in sorted(labels.items())}}
    json.dump(manifest, open(f"{a.outdir}/gate-wording-pairs{a.suffix}.json", "w"), indent=1)
    print("pairs:", len(sel), "of", n_all, dict(counts))
    be = LlamaCppReadout(a.url, rotations=1, domain_hint=DOMAIN)
    for name in a.phrasings.split(","):
        run_phrasing(be, sel, name, not a.no_legend, f"{a.outdir}/gate-wording-{name}{a.suffix}.json",
                     {"pairs_manifest": f"gate-wording-pairs{a.suffix}.json", "n_all_pairs": n_all, "subset": a.subset, "seed": a.seed,
                      "legend_used": not a.no_legend, "model": "Qwen3-4B-Instruct-2507-Q4_0.gguf"})
