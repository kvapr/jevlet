"""Publication figures from results/summary/*.json and the generative run files.

    python -m bench.figures            -> results/figures/{miss-anatomy,router-accuracy,gate-detection,latency}.png
"""

from __future__ import annotations

import collections
import glob
import json
import os
from pathlib import Path

import numpy as np

FIG = Path("results/figures")
SUM = Path("results/summary")
PAL = {"readout": "#3b6ea5", "readout-tiny": "#7fa7d1", "verdict": "#c98a2b", "laya": "#8e6bb3", "deberta": "#5aa469",
       "nli": "#999999", "finetuned": "#c0392b", "generative": "#444444"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _label(backend: str) -> tuple[str, str]:
    b = backend.lower()
    if "0.8b" in b or "tiny" in b:
        return "readout Qwen3.5-0.8B", PAL["readout-tiny"]
    if "readout" in b or "qwen3-4b" in b:
        return "readout Qwen3-4B", PAL["readout"]
    if "verdict" in b:
        return "openJev-verdict 151M (0-shot)", PAL["verdict"]
    if "laya" in b:
        return ("Laya-td 421M (0-shot)" if "typed" in b or "-td" in b else "Laya 421M (0-shot)"), PAL["laya"]
    if "open-jev-deberta" in b or "openjev-deberta" in b:
        return "open-jev-deberta 0.4B (0-shot)", PAL["deberta"]
    if "nli" in b:
        return "NLI cross-encoder (0-shot)", PAL["nli"]
    if b.startswith("ft-"):
        return f"fine-tuned {b[3:]}", PAL["finetuned"]
    return backend, "#666"


def miss_anatomy(run="data/netops/generative_runs/g3-2507-npu-para195.json"):
    from bench import netops
    plt = _plt()
    d = json.load(open(run))
    rows = {r["input"]: r for r in netops.load_rows(("paraphrase",))}
    cats = collections.Counter()
    for rec in d["results"]:
        ok = rec["ok_tool"] and rec["ok_command"] and rec["ok_params_lenient"] and rec["ok_device"]
        if ok:
            cats["correct"] += 1
            continue
        cmd, params = netops._call_of(rec)
        r = rows.get(rec["input"])
        if r is None:
            continue
        if cmd == r["expected_command"]:
            cats["format slip (right command)"] += 1
        elif cmd == "diagnose_port_issues":
            cats["diagnose instead of read"] += 1
        elif netops._is_dangerous(r["expected_command"], cmd):
            cats["dangerous confusion"] += 1
        else:
            cats["other confusion"] += 1
    order = ["correct", "format slip (right command)", "diagnose instead of read", "other confusion", "dangerous confusion"]
    vals = [cats.get(k, 0) for k in order]
    fig, ax = plt.subplots(figsize=(6.4, 2.0), dpi=160)
    colors = ["#5b8c5a", "#9bb7d4", "#d4c29b", "#e0a06a", "#c0392b"]
    left = 0
    for k, v, c in zip(order, vals, colors):
        ax.barh([0], [v], left=left, color=c, edgecolor="white", label=f"{k} ({v})")
        if k == "correct":
            ax.text(left + v / 2, 0, f"correct · {v}", ha="center", va="center", fontsize=8, color="white")
        elif v:
            ax.text(left + v / 2, 0, str(v), ha="center", va="center", fontsize=7, color="#222")
        left += v
    ax.set_yticks([]); ax.set_xlabel("195 paraphrased requests · generative 4B on the NPU · 77.4% end-to-end", fontsize=8)
    ax.set_xlim(0, sum(vals))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "miss-anatomy.png", bbox_inches="tight"); plt.close(fig)
    return dict(cats)


def router_accuracy(files=None):
    plt = _plt()
    files = files or sorted(glob.glob(str(SUM / "router-*.json")))
    rows = []
    for f in files:
        s = json.load(open(f))
        if "questions" not in s:
            continue
        name, color = _label(s["backend"])
        rows.append((name, color, s))
    if not rows:
        return
    qids = ["tool", "command", "is_write", "touches_poe_power"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=150)
    # accuracy per question
    w = 0.8 / len(rows)
    for i, (name, color, s) in enumerate(rows):
        acc = [s["questions"][q]["raw"]["accuracy"] if q in s["questions"] else np.nan for q in qids]
        axes[0].bar(np.arange(len(qids)) + i * w, acc, w, label=name, color=color)
    axes[0].set_xticks(np.arange(len(qids)) + 0.4 - w / 2); axes[0].set_xticklabels(qids, rotation=15)
    axes[0].set_ylim(0, 1.02); axes[0].set_ylabel("accuracy"); axes[0].set_title("router battery (all rows)")
    axes[0].legend(fontsize=6, frameon=False)
    # ECE raw vs scaled for the command question
    names = [r[0] for r in rows]
    raw = [r[2]["questions"]["command"]["raw"]["ece"] for r in rows]
    cal = [r[2]["questions"]["command"]["temp_scaled_oof"]["ece"] for r in rows]
    x = np.arange(len(rows))
    axes[1].bar(x - 0.2, raw, 0.4, label="raw", color="#bbb")
    axes[1].bar(x + 0.2, cal, 0.4, label="temperature-scaled (OOF, grouped)", color="#3b6ea5")
    axes[1].set_xticks(x); axes[1].set_xticklabels([n.split(" (")[0] for n in names], rotation=20, fontsize=7)
    axes[1].set_ylabel("ECE, command question"); axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "router-accuracy.png"); plt.close(fig)


def gate_detection(files=None):
    plt = _plt()
    files = files or sorted(glob.glob(str(SUM / "gate-*.json")))
    rows = []
    for f in files:
        s = json.load(open(f))
        if "auroc" not in s:
            continue
        name, color = _label(s["backend"])
        rk = s.get("real_by_miss_kind", {})
        rows.append((name, color, s["auroc"], s.get("auroc_real_only", np.nan),
                     rk.get("confusion", {}).get("flagged_cal(p<0.5)", 0), rk.get("confusion", {}).get("n", 0),
                     s.get("real_correct", {}).get("false_flag_cal(p<0.5)", 0), s.get("real_correct", {}).get("n", 1),
                     s.get("dangerous_swaps", {}).get("accepted_cal(p>0.5)", 0), s.get("dangerous_swaps", {}).get("n", 1)))
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=150)
    x = np.arange(len(rows)); names = [r[0] for r in rows]; colors = [r[1] for r in rows]
    axes[0].bar(x, [r[2] for r in rows], color=colors); axes[0].set_ylim(0.4, 1); axes[0].set_title("gate AUROC (all pairs)")
    axes[1].bar(x - 0.2, [r[4] / max(r[5], 1) for r in rows], 0.4, color=colors, label="real confusions flagged (p<0.5)")
    axes[1].bar(x + 0.2, [r[6] / max(r[7], 1) for r in rows], 0.4, color="#ccc", label="correct proposals false-flagged")
    axes[1].set_ylim(0, 1); axes[1].legend(fontsize=6, frameon=False); axes[1].set_title("on the generative model's real calls")
    axes[2].bar(x, [r[8] / max(r[9], 1) for r in rows], color=colors); axes[2].set_ylim(0, 1)
    axes[2].set_title("dangerous swaps accepted (p>0.5)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([n.split(" (")[0] for n in names], rotation=25, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "gate-detection.png"); plt.close(fig)


def latency(path="results/latency-cpu.json"):
    if not os.path.exists(path):
        return
    plt = _plt()
    d = json.load(open(path))
    items = sorted(d["results"].items())
    names = [k for k, _ in items]
    v4 = [v["router_4q"]["median_ms"] for _, v in items]
    v1 = [v["noul_1q"]["median_ms"] for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
    x = np.arange(len(names))
    ax.bar(x - 0.2, v4, 0.4, label="router battery (4 questions)", color="#3b6ea5")
    ax.bar(x + 0.2, v1, 0.4, label="one NOUL", color="#9bb7d4")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, fontsize=7); ax.set_ylabel("median ms, CPU"); ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False); ax.set_title(f"CPU latency on {d.get('host','?')}")
    fig.tight_layout(); fig.savefig(FIG / "latency.png"); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    print("miss anatomy:", miss_anatomy())
    router_accuracy()
    gate_detection()
    latency()
    print("figures ->", FIG)
