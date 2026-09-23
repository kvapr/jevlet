"""Figures for the packed-evaluation note, drawn from results/packed-vs-single.json only.

    python -m bench.figures_packed --receipt results/packed-vs-single.json --outdir docs/artifact \
        --prefix packed-evaluation-

Writes <prefix>orders.png (accuracy across all 24 question orders) and <prefix>pack-shape.png
(the battery asked packed as trained vs one question at a time).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QIDS = ["tool", "command", "is_write", "touches_poe_power"]
LABEL = {"tool": "tool (8-way)", "command": "command (24-way)", "is_write": "is_write",
         "touches_poe_power": "touches_poe_power"}
ROUND = {"round1": ("round 1, fixed order", "#c98a2b"), "round2": ("round 2, shuffled", "#a8404b")}
INK, MUTED, RULE = "#172026", "#5e6b72", "#d3dbd8"


def _frame(ax):
    ax.set_facecolor("white")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=RULE, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def fig_orders(rec, out):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor("white")
    ylab, ypos = [], []
    y = 0.0
    for qid in QIDS:
        for tag in ("round2", "round1"):
            b = rec["bundles"][tag]
            accs = [o["per_question"][qid]["acc"] for o in b["orders"].values()]
            o = b["by_slot"][qid]["over_orders"]
            name, color = ROUND[tag]
            ax.plot([o["acc_min"], o["acc_max"]], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=2)
            ax.scatter(accs, [y] * len(accs), s=26, color=color, alpha=0.45, linewidths=0, zorder=3)
            ax.scatter([o["acc_mean"]], [y], s=54, color=color, marker="D",
                       edgecolors="white", linewidths=1.4, zorder=4)
            span = (o["acc_max"] - o["acc_min"]) * 100
            ax.text(1.005, y, f"{o['acc_mean']*100:.1f}%  span {span:.0f} pts", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8.2, color=MUTED, family="monospace")
            ylab.append(name)
            ypos.append(y)
            y -= 1.0
        y -= 0.65
    _frame(ax)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=9, color=INK)
    ax.set_ylim(y + 0.45, 1.0)
    for i, qid in enumerate(QIDS):
        yq = -(i * 2.65) + 0.62
        ax.text(0.005, yq, LABEL[qid], transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=9.5, color=INK, fontweight="semibold")
    ax.set_title("Accuracy on the 126 held-out rows, one point per question order (24 orders)",
                 fontsize=10, color=INK, loc="left", pad=10)
    fig.subplots_adjust(left=0.20, right=0.80, top=0.92, bottom=0.08)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


def fig_pack_shape(rec, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=150, sharey=True)
    fig.patch.set_facecolor("white")
    titles = ["asked packed, all four questions\n(the layout it was trained on)",
              "asked alone, one question per pass\n(the whole pack is that question)"]
    for ax, title, cond in zip(axes, titles, ("packed", "alone")):
        for i, qid in enumerate(QIDS):
            base = len(QIDS) - 1 - i
            for j, tag in enumerate(("round1", "round2")):
                b = rec["bundles"][tag]
                key = "packed_trained_order" if cond == "packed" else f"alone_{qid}"
                acc = b["conditions"][key]["per_question"][qid]["acc"]
                name, color = ROUND[tag]
                yy = base + (0.19 if j == 0 else -0.19)
                ax.barh(yy, acc, height=0.34, color=color, zorder=3,
                        label=name if i == 0 else None)
                ax.text(acc + 0.02, yy, f"{acc*100:.1f}%", va="center", ha="left",
                        fontsize=8.2, color=INK, family="monospace", zorder=4)
        _frame(ax)
        ax.set_xlim(0, 1.18)
        ax.set_ylim(-0.6, len(QIDS) - 0.4)
        ax.set_yticks(range(len(QIDS)))
        ax.set_yticklabels([LABEL[q] for q in reversed(QIDS)], fontsize=9, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               labelcolor=INK, handlelength=1.1, bbox_to_anchor=(0.55, -0.01))
    fig.subplots_adjust(left=0.19, right=0.99, top=0.79, bottom=0.21, wspace=0.30)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", default="results/packed-vs-single.json")
    ap.add_argument("--outdir", default="docs/artifact")
    ap.add_argument("--prefix", default="packed-evaluation-")
    a = ap.parse_args()
    rec = json.load(open(a.receipt))
    d = Path(a.outdir)
    d.mkdir(parents=True, exist_ok=True)
    fig_orders(rec, d / f"{a.prefix}orders.png")
    fig_pack_shape(rec, d / f"{a.prefix}pack-shape.png")
