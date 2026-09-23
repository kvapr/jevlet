"""Operating-point analysis of a gate-*.json result: where the confirm-card thresholds land.

Reads a gate readout file (default: the study's results/gate-qwen3-4b-q4_0-cpu.json), refits the
out-of-fold temperature exactly as bench.analyze does (5 folds grouped by seed request) and
reports what the card thresholds of jevlet.integrations.lance_local (ok at p >= 0.85, mismatch at
p <= 0.4, "check" between) do per pair kind and per action class, the risk-coverage curve those
thresholds are one point on, and a few concrete pairs with their probabilities for a worked card.

    python -m bench.gate_operating_point results/gate-qwen3-4b-q4_0-cpu.json
    -> results/gate-operating-point-<tag>.json
       docs/artifact/tutorial-confirm-gate-risk-coverage.png, tutorial-confirm-gate-buckets.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from jevlet import metrics as M
from jevlet.calibrate import cv_temperature
from jevlet.integrations.lance_local import OP_CLASS, annotate_proposal  # noqa: F401 (thresholds documented here)
from bench.analyze import _relabel_real

OK_ABOVE, MISMATCH_BELOW = 0.85, 0.4          # the defaults of annotate_proposal()


def buckets(pv):
    ok = pv >= OK_ABOVE
    mis = pv <= MISMATCH_BELOW
    chk = ~ok & ~mis
    n = max(len(pv), 1)
    return {"n": int(len(pv)), "mismatch": int(mis.sum()), "check": int(chk.sum()), "ok": int(ok.sum()),
            "frac_mismatch": float(mis.sum() / n), "frac_check": float(chk.sum() / n), "frac_ok": float(ok.sum() / n)}


def action_class(command: str) -> str:
    if command in OP_CLASS:
        return OP_CLASS[command]
    return "read" if command.startswith("show_") else command


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default="results/gate-qwen3-4b-q4_0-cpu.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figdir", default="docs/artifact")
    ap.add_argument("--prefix", default="tutorial-confirm-gate")
    a = ap.parse_args()

    d = json.load(open(a.file))
    res = _relabel_real(d["results"])
    p = np.array([x["p_yes"] for x in res], dtype=float)
    y = np.array([x["label"] for x in res])
    kinds = np.array([x["kind"] for x in res])
    miss = np.array([x.get("miss_kind", "-") for x in res])
    cmds = [x["command"] for x in res]
    groups = [x["base_input"] for x in res]
    dang = np.array([bool(x.get("dangerous")) for x in res])
    L = np.stack([np.log(np.clip(1 - p, 1e-9, 1)), np.log(np.clip(p, 1e-9, 1))], 1)
    Pc, temps = cv_temperature(L, y, groups, n_folds=5)
    pc = Pc[:, 1]
    correct_raw, correct_cal = (p > 0.5) == (y == 1), (pc > 0.5) == (y == 1)

    out = {"file": os.path.basename(a.file), "backend": d["meta"].get("backend"), "n": int(len(res)),
           "thresholds": {"ok_above": OK_ABOVE, "mismatch_below": MISMATCH_BELOW},
           "temperatures": [float(t) for t in temps],
           "auroc_raw": M.auroc(p, y), "auroc_cal": M.auroc(pc, y),
           "ece_raw": M.ece(np.maximum(p, 1 - p), correct_raw), "ece_cal": M.ece(np.maximum(pc, 1 - pc), correct_cal),
           "acc_at_0.5_raw": float(correct_raw.mean()), "acc_at_0.5_cal": float(correct_cal.mean())}

    # --- card buckets per pair kind (and the real pairs by miss kind), raw and calibrated
    out["buckets_cal"], out["buckets_raw"] = {}, {}
    sel = {"gold": kinds == "gold", "real_correct": (kinds == "real") & (y == 1), "real_miss": (kinds == "real") & (y == 0),
           "swap_op": kinds == "swap_op", "wrong_port": kinds == "wrong_port", "dangerous_swap": dang & (y == 0)}
    for k in sorted(set(miss[kinds == "real"])):
        sel[f"real:{k}"] = (kinds == "real") & (miss == k)
    for k, m in sel.items():
        out["buckets_cal"][k] = buckets(pc[m])
        out["buckets_raw"][k] = buckets(p[m])

    # --- per action class of the PROPOSED command, calibrated
    out["by_action_class_cal"] = {}
    cls = np.array([action_class(c) for c in cmds])
    for c in sorted(set(cls)):
        m = cls == c
        pos, neg = m & (y == 1), m & (y == 0)
        out["by_action_class_cal"][c] = {
            "n": int(m.sum()), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
            "neg_accepted(p>0.5)": int((pc[neg] > 0.5).sum()), "neg_ok(p>=0.85)": int((pc[neg] >= OK_ABOVE).sum()),
            "neg_mismatch(p<=0.4)": int((pc[neg] <= MISMATCH_BELOW).sum()),
            "pos_flagged(p<0.5)": int((pc[pos] < 0.5).sum()), "pos_mismatch(p<=0.4)": int((pc[pos] <= MISMATCH_BELOW).sum()),
            "pos_ok(p>=0.85)": int((pc[pos] >= OK_ABOVE).sum()),
            "mean_p_pos": float(pc[pos].mean()) if pos.any() else None, "mean_p_neg": float(pc[neg].mean()) if neg.any() else None}

    # --- threshold sweeps, calibrated
    pos, neg = y == 1, y == 0
    out["ok_line_sweep_cal"] = [{"tau": t, "pos_shown_ok": float((pc[pos] >= t).mean()), "neg_shown_ok": float((pc[neg] >= t).mean()),
                                 "n_neg_shown_ok": int((pc[neg] >= t).sum())} for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)]
    out["mismatch_line_sweep_cal"] = [{"tau": t, "neg_shown_mismatch": float((pc[neg] <= t).mean()), "pos_shown_mismatch": float((pc[pos] <= t).mean()),
                                       "n_pos_shown_mismatch": int((pc[pos] <= t).sum())} for t in (0.2, 0.3, 0.4, 0.5)]

    # --- risk-coverage on calibrated confidence (same construction as bench.analyze's figure)
    conf = np.maximum(pc, 1 - pc)
    order = np.argsort(-conf)
    cov = np.arange(1, len(order) + 1) / len(order)
    risk = np.cumsum(~correct_cal[order]) / np.arange(1, len(order) + 1)
    out["aurc_cal"] = M.aurc(conf, correct_cal)
    out["risk_coverage_cal"] = []
    for c in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        i = int(np.searchsorted(cov, c, side="left"))
        i = min(i, len(cov) - 1)
        out["risk_coverage_cal"].append({"coverage": float(cov[i]), "risk": float(risk[i]), "conf_threshold": float(conf[order][i]),
                                         "n_covered": int(i + 1), "n_wrong": int(round(risk[i] * (i + 1)))})
    out["selective_cal"] = M.selective(conf, correct_cal, thresholds=(0.6, 0.7, 0.85, 0.9, 0.95))
    out["selective_raw"] = M.selective(np.maximum(p, 1 - p), correct_raw, thresholds=(0.6, 0.7, 0.85, 0.9, 0.95))
    decided = (pc >= OK_ABOVE) | (pc <= MISMATCH_BELOW)
    out["card_band_as_point"] = {"coverage": float(decided.mean()), "risk": float((~correct_cal[decided]).mean()),
                                 "n_decided": int(decided.sum()), "n_wrong": int((~correct_cal[decided]).sum()),
                                 "n_check": int((~decided).sum()), "check_error_if_forced": float((~correct_cal[~decided]).mean())}

    # --- concrete pairs for a worked card: one seed with gold / swap / wrong-port (+ real if present)
    out["examples"] = []
    for i, x in enumerate(res):
        if x["kind"] == "gold" and x["true_command"] == "configure_poe_mode" and x.get("style", "canon") == "canon" \
                and (x.get("params") or {}).get("mode") == "disable":
            seed = x["input"]
            for j, z in enumerate(res):
                if z["input"] == seed:
                    out["examples"].append({"input": z["input"], "kind": z["kind"], "command": z["command"], "params": z["params"],
                                            "label": int(z["label"]), "dangerous": bool(z.get("dangerous")),
                                            "p_raw": float(p[j]), "p_cal": float(pc[j]), "label_mass": z.get("label_mass")})
            break
    # a real dangerous confusion the agent actually made, with its probabilities
    for j, z in enumerate(res):
        if z["kind"] == "real" and z.get("dangerous") and z["label"] == 0:
            out["examples"].append({"input": z["input"], "kind": "real", "miss_kind": z.get("miss_kind"), "command": z["command"],
                                    "params": z["params"], "true_command": z["true_command"], "label": 0, "dangerous": True,
                                    "p_raw": float(p[j]), "p_cal": float(pc[j])})
            break

    tag = os.path.basename(a.file)[len("gate-"):-5]
    outp = a.out or f"results/gate-operating-point-{tag}.json"
    json.dump(out, open(outp, "w"), indent=1)
    print("wrote", outp)
    figures(out, cov, risk, a.figdir, a.prefix)


def figures(out, cov, risk, figdir, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(figdir).mkdir(parents=True, exist_ok=True)
    ink, muted, rule, accent = "#172026", "#5e6b72", "#d3dbd8", "#2f6690"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": rule, "axes.labelcolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    # 1. risk-coverage with the card band marked
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150, facecolor="white")
    ax.plot(cov, risk, color=accent, lw=2)
    pt = out["card_band_as_point"]
    ax.plot([pt["coverage"]], [pt["risk"]], "o", ms=8, mfc="white", mec=accent, mew=2)
    ax.annotate(f'card lines 0.85 / 0.4: decides {pt["coverage"]*100:.0f}% of pairs, {pt["risk"]*100:.1f}% wrong',
                (pt["coverage"], pt["risk"]), xytext=(pt["coverage"] - 0.02, pt["risk"] + 0.045), ha="right", color=ink, fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.8})
    r70 = next(r for r in out["risk_coverage_cal"] if abs(r["coverage"] - 0.7) < 0.01)
    ax.plot([r70["coverage"]], [r70["risk"]], "o", ms=6, color=accent)
    ax.annotate(f'70% coverage: {r70["risk"]*100:.1f}% wrong', (r70["coverage"], r70["risk"]),
                xytext=(r70["coverage"] - 0.02, r70["risk"] + 0.06), ha="right", color=ink, fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.8})
    ax.set_xlim(0, 1.0); ax.set_ylim(0, max(risk.max() * 1.25, 0.2))
    ax.set_xlabel("coverage: share of pairs the gate decides on (most confident first)")
    ax.set_ylabel("error rate among decided pairs")
    ax.grid(axis="y", color=rule, lw=0.6); ax.set_axisbelow(True)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0]); ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    ax.set_yticks([0, .05, .1, .15, .2]); ax.set_yticklabels(["0", "5%", "10%", "15%", "20%"])
    fig.tight_layout(); fig.savefig(Path(figdir) / f"{prefix}-risk-coverage.png", facecolor="white"); plt.close(fig)
    # 2. card buckets per pair kind
    rows = [("gold call", "gold"), ("agent's call, correct", "real_correct"), ("agent's call, a miss", "real_miss"),
            ("swapped operation", "swap_op"), ("wrong port", "wrong_port")]
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150, facecolor="white")
    cols = {"mismatch": "#b5432f", "check": "#c7831c", "ok": "#2e7d5b"}
    for r, (label, k) in enumerate(rows):
        b = out["buckets_cal"][k]
        left = 0.0
        for seg in ("mismatch", "check", "ok"):
            f = b[f"frac_{seg}"]
            ax.barh(r, f, left=left, color=cols[seg], height=0.62, edgecolor="white", linewidth=2,
                    label=f'"{seg}" line' if r == 0 else None)
            if f >= 0.09:
                ax.text(left + f / 2, r, f"{f*100:.0f}%", ha="center", va="center", color="white", fontsize=8.5)
            left += f
        ax.text(1.015, r, f"n = {b['n']}", va="center", color=muted, fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for l, _ in rows], color=ink); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3, frameon=False, fontsize=8.5)
    ax.set_xlabel("share of pairs, calibrated p(matches request), thresholds 0.4 / 0.85")
    fig.tight_layout(); fig.savefig(Path(figdir) / f"{prefix}-buckets.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figures ->", figdir)


if __name__ == "__main__":
    main()
