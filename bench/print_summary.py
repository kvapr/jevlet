"""Print the key numbers of results/summary/*.json compactly.  python -m bench.print_summary [names...]"""

from __future__ import annotations

import glob
import json
import sys

files = sys.argv[1:] or sorted(glob.glob("results/summary/*.json"))
for f in files:
    s = json.load(open(f))
    print("=====", s.get("backend"), "|", s.get("file"))
    if "questions" in s:
        for qid, r in s["questions"].items():
            raw, cal = r["raw"], r["temp_scaled_oof"]
            sel_raw = {x["tau"]: (x["n"], x["wrong"]) for x in raw["selective_jev"]}
            sel_cal = {x["tau"]: (x["n"], x["wrong"]) for x in cal["selective_jev"]}
            print(f"  {qid:18s} n={r['n']:4d} acc={raw['accuracy']:.3f} alt_ok={r.get('acc_allowing_alt_ok', float('nan')):.3f} "
                  f"ECE {raw['ece']:.3f}->{cal['ece']:.3f} NLL {raw['nll']:.2f}->{cal['nll']:.2f} Brier {raw['brier']:.3f}->{cal['brier']:.3f} "
                  f"flip={r.get('order_flip_rate', float('nan')):.3f} single_rot={r.get('acc_single_rotation', float('nan')):.3f} "
                  f"T={[round(t, 2) for t in r.get('temperatures', [])]}")
            print("      style:", {k: round(v["acc"], 3) for k, v in r["by_style"].items()},
                  "| cov/wrong @0.9 raw", sel_raw.get(0.9), "cal", sel_cal.get(0.9), "| @0.5 raw", sel_raw.get(0.5), "cal", sel_cal.get(0.5))
            if qid == "command":
                print("      confusions:", [(c["true"], c["pred"], c["n"]) for c in r["top_confusions"][:8]],
                      "| dangerous:", r["dangerous_confusions"], r["dangerous_confusions_conf_gt_0.9"])
            if "latency_s" in r:
                print("      latency ms: median %.0f p95 %.0f" % (r["latency_s"]["median"] * 1000, r["latency_s"]["p95"] * 1000),
                      "| label_mass", round(r.get("label_mass_mean", float("nan")), 3))
        if "call_latency_s" in s:
            print("  call latency ms: median %.0f p95 %.0f" % (s["call_latency_s"]["median"] * 1000, s["call_latency_s"]["p95"] * 1000))
    else:
        print(f"  gate n={s['n']} AUROC={s['auroc']:.3f} acc@0.5 {s['acc_at_0.5_raw']:.3f}->{s['acc_at_0.5_temp_scaled']:.3f} "
              f"ECE {s['ece_raw']:.3f}->{s['ece_temp_scaled']:.3f} Brier {s['brier_raw']:.3f}->{s['brier_temp_scaled']:.3f} T={[round(t,2) for t in s['temperatures']]}")
        for k, v in s["by_kind"].items():
            print(f"      {k:11s} n={v['n']:4d} mean_p raw {v['mean_p_yes_raw']:.3f} cal {v['mean_p_yes_cal']:.3f} | accepted raw {v['frac_accepted_raw(p>0.5)']:.3f} cal {v['frac_accepted_cal(p>0.5)']:.3f}")
        if "real_misses" in s:
            print("      real misses:", s["real_misses"], "| real correct:", s["real_correct"], "| AUROC real-only %.3f" % s["auroc_real_only"])
        if "real_by_miss_kind" in s:
            print("      real by miss kind:", {k: (v["n"], round(v["label_mean"], 2), v["flagged_cal(p<0.5)"], v["flagged_cal(p<0.9)"]) for k, v in s["real_by_miss_kind"].items()}, "(n, label mean, flagged<.5, flagged<.9)")
        if "dangerous_swaps" in s:
            print("      dangerous swaps:", s["dangerous_swaps"])
        print("      by style AUROC:", {k: round(v["auroc"], 3) for k, v in s["by_style"].items()})
        if "latency_s" in s:
            print("      latency ms: median %.0f" % (s["latency_s"]["median"] * 1000))
