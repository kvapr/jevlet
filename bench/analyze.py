"""Analysis of router-*.json and gate-*.json result files: accuracy, calibration (raw and
out-of-fold temperature-scaled, grouped by seed request), selective accuracy at the Jev
thresholds, confusable-pair confusions, order sensitivity, latency; plus figures.

    python -m bench.analyze results/router-qwen3-4b-q4_0-cpu-rot3.json results/gate-qwen3-4b-q4_0-cpu.json ...
    -> results/summary/<name>.json, results/summary/<name>.md, results/figures/*.png
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

from jevlet import metrics as M
from jevlet.calibrate import cv_temperature, softmax, group_folds
from bench.netops import NONE, DANGEROUS_PAIRS

FIG = Path("results/figures")
SUM = Path("results/summary")


# ------------------------------------------------------------------ router
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
        elif qid == "ambiguity":
            y.append(int(r.get("ambiguity") or 0))
        else:
            y.append(-1)
    return np.array(y)


def _alt_ok_mask(qid, q, rows, pred):
    """A prediction counts as acceptable if it is the gold or one of the row's alt_ok commands."""
    if qid != "command":
        return None
    opts = q["options"]
    ok = []
    for r, p in zip(rows, pred):
        alts = set(r.get("alt_ok") or [])
        ok.append(opts[p] in alts)
    return np.array(ok)


GROUP_FILTER: set | None = None   # set from --groups: restrict rows to these base_inputs (held-out groups)


def analyze_router(path):
    d = json.load(open(path))
    meta, res = d["meta"], d["results"]
    rows = meta["rows"]
    if GROUP_FILTER is not None:
        keep = [i for i, r in enumerate(rows) if r.get("base_input") in GROUP_FILTER]
        rows = [rows[i] for i in keep]
        idx = {old: new for new, old in enumerate(keep)}
        res = {qid: [dict(x, i=idx[x["i"]]) for x in recs if x["i"] in idx] for qid, recs in res.items()}
    groups = [r["base_input"] for r in rows]
    out = {"file": os.path.basename(path), "backend": meta.get("backend"), "n": len(rows), "questions": {}}
    styles = np.array([r.get("style") or "canon" for r in rows])
    splits = np.array([r.get("split") or "?" for r in rows])
    for qid, q in meta["questions"].items():
        recs = sorted(res[qid], key=lambda x: x["i"])
        P = np.array([x["probs"] for x in recs], dtype=float)
        L = np.array([x["logits"] if x.get("logits") else np.log(np.clip(x["probs"], 1e-9, 1)) for x in recs], dtype=float)
        y = _labels_for(qid, q, rows)
        valid = y >= 0
        if qid == "ambiguity":
            valid &= np.array([r.get("split") not in ("canonical", "paraphrase") for r in rows])
        P, L, y = P[valid], L[valid], y[valid]
        g = [groups[i] for i in np.where(valid)[0]]
        st, sp = styles[valid], splits[valid]
        pred = P.argmax(1)
        correct = pred == y
        r = {"n": int(valid.sum()), "raw": M.summarize(P, y)}
        # out-of-fold temperature scaling grouped by seed request
        Pc, temps = cv_temperature(L, y, g, n_folds=5)
        r["temp_scaled_oof"] = M.summarize(Pc, y)
        r["temperatures"] = temps
        r["by_style"] = {s: {"n": int((st == s).sum()), "acc": float(correct[st == s].mean())}
                         for s in sorted(set(st))}
        r["by_split"] = {s: {"n": int((sp == s).sum()), "acc": float(correct[sp == s].mean())}
                         for s in sorted(set(sp))}
        alt = _alt_ok_mask(qid, q, [rows[i] for i in np.where(valid)[0]], pred)
        if alt is not None:
            r["acc_allowing_alt_ok"] = float((correct | alt).mean())
        # confusable pairs (command question)
        if qid == "command":
            opts = q["options"]
            conf = collections.Counter()
            for t, p in zip(y, pred):
                if t != p:
                    conf[(opts[t], opts[p])] += 1
            r["top_confusions"] = [{"true": a, "pred": b, "n": n} for (a, b), n in conf.most_common(12)]
            dang = [(opts[t], opts[p]) for t, p in zip(y, pred) if t != p and any({opts[t], opts[p]} == s for s in DANGEROUS_PAIRS)]
            r["dangerous_confusions"] = len(dang)
            # wrong-and-confident: dangerous confusions the raw / scaled confidence would have waved through
            cj_raw, cj_cal = M.jev_confidence(P), M.jev_confidence(Pc)
            dmask = np.array([t != p and any({opts[t], opts[p]} == s for s in DANGEROUS_PAIRS) for t, p in zip(y, pred)])
            r["dangerous_confusions_conf_gt_0.9"] = {"raw": int((cj_raw[dmask] > 0.9).sum()), "temp_scaled": int((cj_cal[dmask] > 0.9).sum())}
        # order sensitivity when rotations were recorded
        if recs and recs[0].get("per_rotation"):
            flips = 0
            tot = 0
            for x in recs:
                pr = [int(np.argmax(v)) for v in x["per_rotation"]]
                tot += 1
                flips += len(set(pr)) > 1
            r["order_flip_rate"] = flips / tot
            # accuracy of a single rotation (the first) vs averaged
            P0 = np.array([x["per_rotation"][0] for x in recs], dtype=float)[valid]
            r["acc_single_rotation"] = float((P0.argmax(1) == y).mean())
        lat = [x.get("latency_s") for x in recs if x.get("latency_s") is not None]
        if lat:
            r["latency_s"] = {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "mean": float(np.mean(lat))}
        lm = [x.get("label_mass") for x in recs if x.get("label_mass") is not None]
        if lm:
            r["label_mass_mean"] = float(np.mean(lm))
        out["questions"][qid] = r
        _reliability_fig(P, Pc, y, f"{out['backend']}-{qid}")
    if meta.get("call_latency_s"):
        cl = meta["call_latency_s"]
        out["call_latency_s"] = {"median": float(np.median(cl)), "p95": float(np.percentile(cl, 95))}
    return out


# ------------------------------------------------------------------ gate
def _relabel_real(res):
    """Recompute the SEMANTIC gate label and miss kind of 'real' pairs from the dataset, so gate
    result files produced before the labeling fix are analyzed consistently (a bare-command
    format slip with the right command and params DOES match the request)."""
    from bench import netops
    rows = {r["input"]: r for r in netops.load_rows(("canonical", "paraphrase"))}
    for x in res:
        if x.get("kind") != "real":
            continue
        r = rows.get(x["input"])
        if r is None:
            continue
        exp = r.get("expected_params") or {}
        params = x.get("params") or {}
        params_ok = all(str(params.get(k)) == str(v) for k, v in exp.items())
        same_cmd = x["command"] == r["expected_command"]
        e2e = x.get("e2e", x["label"])
        x["label"] = int(same_cmd and params_ok)
        x["miss_kind"] = ("none" if e2e else "format_only" if (same_cmd and params_ok) else "params" if same_cmd
                          else "diagnose_instead" if x["command"] == "diagnose_port_issues" else "confusion")
    return res


def analyze_gate(path):
    d = json.load(open(path))
    res = _relabel_real(d["results"])
    if GROUP_FILTER is not None:
        res = [x for x in res if x.get("base_input") in GROUP_FILTER]
    p = np.array([x["p_yes"] for x in res], dtype=float)
    y = np.array([x["label"] for x in res])
    kinds = np.array([x["kind"] for x in res])
    styles = np.array([x.get("style") or "canon" for x in res])
    dang = np.array([bool(x.get("dangerous")) for x in res])
    groups = [x["base_input"] for x in res]
    L = np.stack([np.log(np.clip(1 - p, 1e-9, 1)), np.log(np.clip(p, 1e-9, 1))], 1)
    Pc, temps = cv_temperature(L, y, groups, n_folds=5)
    pc = Pc[:, 1]
    out = {"file": os.path.basename(path), "backend": d["meta"].get("backend"), "n": len(res),
           "auroc": M.auroc(p, y), "acc_at_0.5_raw": float(((p > 0.5) == (y == 1)).mean()),
           "acc_at_0.5_temp_scaled": float(((pc > 0.5) == (y == 1)).mean()),
           "ece_raw": M.ece(np.maximum(p, 1 - p), (p > 0.5) == (y == 1)),
           "ece_temp_scaled": M.ece(np.maximum(pc, 1 - pc), (pc > 0.5) == (y == 1)),
           "brier_raw": float(((p - y) ** 2).mean()), "brier_temp_scaled": float(((pc - y) ** 2).mean()),
           "temperatures": temps, "by_kind": {}, "by_style": {}}
    for k in sorted(set(kinds)):
        m = kinds == k
        out["by_kind"][k] = {"n": int(m.sum()), "mean_p_yes_raw": float(p[m].mean()), "mean_p_yes_cal": float(pc[m].mean()),
                             "frac_accepted_raw(p>0.5)": float((p[m] > 0.5).mean()),
                             "frac_accepted_cal(p>0.5)": float((pc[m] > 0.5).mean())}
    for s in sorted(set(styles)):
        m = styles == s
        out["by_style"][s] = {"n": int(m.sum()), "auroc": M.auroc(p[m], y[m])}
    # the product question: of the generative model's REAL misses, how many does the gate flag?
    real = kinds == "real"
    if real.any():
        rm = real & (y == 0)
        out["real_misses"] = {"n": int(rm.sum()),
                              "flagged_raw(p<0.5)": int((p[rm] < 0.5).sum()), "flagged_cal(p<0.5)": int((pc[rm] < 0.5).sum()),
                              "flagged_cal(p<0.9)": int((pc[rm] < 0.9).sum())}
        rc = real & (y == 1)
        out["real_correct"] = {"n": int(rc.sum()), "false_flag_raw(p<0.5)": int((p[rc] < 0.5).sum()),
                               "false_flag_cal(p<0.5)": int((pc[rc] < 0.5).sum()), "false_flag_cal(p<0.9)": int((pc[rc] < 0.9).sum())}
        out["auroc_real_only"] = M.auroc(p[real], y[real])
        mk = np.array([x.get("miss_kind", "?") for x in res])
        out["real_by_miss_kind"] = {}
        for k in sorted(set(mk[real])):
            m = real & (mk == k)
            out["real_by_miss_kind"][k] = {"n": int(m.sum()), "label_mean": float(y[m].mean()),
                                           "flagged_cal(p<0.5)": int((pc[m] < 0.5).sum()), "flagged_cal(p<0.9)": int((pc[m] < 0.9).sum()),
                                           "mean_p_cal": float(pc[m].mean())}
    if dang.any():
        dm = dang & (y == 0)
        out["dangerous_swaps"] = {"n": int(dm.sum()), "accepted_raw(p>0.5)": int((p[dm] > 0.5).sum()),
                                  "accepted_cal(p>0.5)": int((pc[dm] > 0.5).sum()), "accepted_cal(p>0.9)": int((pc[dm] > 0.9).sum())}
    lat = [x.get("latency_s") for x in res if x.get("latency_s") is not None]
    if lat:
        out["latency_s"] = {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))}
    _risk_coverage_fig(pc, y, f"{out['backend']}-gate")
    return out


# ------------------------------------------------------------------ figures
def _reliability_fig(P, Pc, y, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), dpi=130)
    for ax, PP, title in ((axes[0], P, "raw"), (axes[1], Pc, "temperature-scaled (OOF)")):
        conf, corr = PP.max(1), PP.argmax(1) == y
        tab = M.reliability_table(conf, corr, 10)
        xs = [(i + 0.5) / 10 for i in range(10)]
        accs = [t["acc"] if t["acc"] is not None else np.nan for t in tab]
        ns = [t["n"] for t in tab]
        ax.bar(xs, accs, width=0.1, color="#5b7fa6", edgecolor="white")
        ax.plot([0, 1], [0, 1], "--", color="#999")
        for x, n, a in zip(xs, ns, accs):
            if n:
                ax.text(x, (a if not np.isnan(a) else 0) + 0.02, str(n), ha="center", fontsize=6, color="#333")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel("confidence (max p)"); ax.set_ylabel("accuracy")
        ax.set_title(f"{title}  ECE={M.ece(conf, corr):.3f}", fontsize=9)
    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"reliability-{name}.png")
    plt.close(fig)


def _risk_coverage_fig(pc, y, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    conf = np.maximum(pc, 1 - pc)
    correct = (pc > 0.5) == (y == 1)
    order = np.argsort(-conf)
    cov = np.arange(1, len(order) + 1) / len(order)
    risk = np.cumsum(~correct[order]) / np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=130)
    ax.plot(cov, risk, color="#5b7fa6")
    ax.set_xlabel("coverage"); ax.set_ylabel("error rate among covered"); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / f"risk-coverage-{name}.png"); plt.close(fig)


# ------------------------------------------------------------------ report
def to_markdown(summaries):
    lines = ["| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        if "questions" in s:
            for qid, r in s["questions"].items():
                raw, cal = r["raw"], r["temp_scaled_oof"]
                w_raw = next(x["wrong"] for x in raw["selective_jev"] if x["tau"] == 0.9)
                w_cal = next(x["wrong"] for x in cal["selective_jev"] if x["tau"] == 0.9)
                lat = r.get("latency_s", {}).get("median")
                lines.append(f"| {s['backend']} | {qid} | {r['n']} | {raw['accuracy']:.3f} | "
                             f"{r.get('acc_allowing_alt_ok', float('nan')):.3f} | {raw['ece']:.3f} | {cal['ece']:.3f} | {raw['brier']:.3f} | "
                             f"{w_raw}→{w_cal} | {r.get('order_flip_rate', float('nan')):.2f} | "
                             f"{(lat*1000):.0f} ms |" if lat is not None else
                             f"| {s['backend']} | {qid} | {r['n']} | {raw['accuracy']:.3f} | "
                             f"{r.get('acc_allowing_alt_ok', float('nan')):.3f} | {raw['ece']:.3f} | {cal['ece']:.3f} | {raw['brier']:.3f} | "
                             f"{w_raw}→{w_cal} | {r.get('order_flip_rate', float('nan')):.2f} | n/a |")
        else:
            rm, dg = s.get("real_misses", {}), s.get("dangerous_swaps", {})
            lines.append(f"| {s['backend']} | gate | {s['n']} | AUROC {s['auroc']:.3f} | acc@.5 {s['acc_at_0.5_raw']:.3f}→{s['acc_at_0.5_temp_scaled']:.3f} | "
                         f"{s['ece_raw']:.3f} | {s['ece_temp_scaled']:.3f} | {s['brier_raw']:.3f} | "
                         f"real misses flagged {rm.get('flagged_cal(p<0.5)','-')}/{rm.get('n','-')} | dangerous accepted {dg.get('accepted_cal(p>0.5)','-')}/{dg.get('n','-')} | "
                         f"{(s.get('latency_s',{}).get('median') or 0)*1000:.0f} ms |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", default="summary")
    ap.add_argument("--groups", default=None, help="split_manifest.json: restrict to its test_groups (held-out comparison)")
    a = ap.parse_args()
    SUM.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if a.groups:
        GROUP_FILTER = set(json.load(open(a.groups))["test_groups"])
        suffix = "-heldout"
    sums = []
    for f in a.files:
        s = analyze_gate(f) if os.path.basename(f).startswith("gate-") else analyze_router(f)
        s["file"] = s["file"][:-5] + suffix + ".json"
        sums.append(s)
        json.dump(s, open(SUM / (os.path.basename(f)[:-5] + suffix + ".json"), "w"), indent=1)
    md = to_markdown(sums)
    open(SUM / f"{a.name}.md", "w").write(md + "\n")
    print(md)
