"""The generative tool-calling baseline, scored on the same rows and axes as the decision layer:
tool selection, command (semantic: a bare-command format slip with the right command counts),
end-to-end as graded by the original harness, per style.

    python -m bench.generative_baseline data/netops/generative_runs/g3-2507-npu-para195.json [more runs...]
"""

from __future__ import annotations

import collections
import json
import sys

from bench import netops


def score(run_file):
    d = json.load(open(run_file))
    rows = {r["input"]: r for r in netops.load_rows(("canonical", "paraphrase"))}
    by = collections.defaultdict(lambda: collections.Counter())
    for rec in d["results"]:
        r = rows.get(rec["input"])
        if r is None:
            continue
        st = r.get("style", "canon")
        cmd, params = netops._call_of(rec)
        e2e = bool(rec.get("ok_tool") and rec.get("ok_command") and rec.get("ok_params_lenient") and rec.get("ok_device"))
        tool_ok = bool(rec.get("ok_tool")) or (cmd == r["expected_command"])   # bare right command = right tool family
        cmd_ok = cmd == r["expected_command"]
        for k in (st, "all"):
            by[k]["n"] += 1
            by[k]["e2e"] += e2e
            by[k]["tool"] += tool_ok
            by[k]["command"] += cmd_ok
    return {k: {"n": v["n"], "e2e": v["e2e"] / v["n"], "tool_or_bare_cmd": v["tool"] / v["n"], "command_semantic": v["command"] / v["n"]}
            for k, v in by.items()}, d.get("summary", {})


if __name__ == "__main__":
    for f in sys.argv[1:]:
        s, summ = score(f)
        print("=====", f, "| model:", summ.get("model"), "| median latency s:", summ.get("median_latency_s"))
        for k in sorted(s):
            v = s[k]
            print(f"  {k:6s} n={v['n']:3d} e2e={v['e2e']:.3f} tool={v['tool_or_bare_cmd']:.3f} command={v['command_semantic']:.3f}")
