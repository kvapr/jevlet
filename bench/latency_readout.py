"""Quiet-host latency of the llama-server readout: warm prefix cache, one client, no other load.
Reports per-question wall time and the server's own prompt_n / prompt_ms (tokens actually evaluated).

    python -m bench.latency_readout --url http://127.0.0.1:18999 --tag qwen3-4b --n 30
"""

from __future__ import annotations

import argparse
import json
import statistics

from jevlet.backends.llamacpp import LlamaCppReadout
from jevlet.schema import normalize_questions
from bench import netops
from bench.run_readout import DOMAIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18999")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="results/latency-readout.json")
    a = ap.parse_args()
    be = LlamaCppReadout(a.url, rotations=1, domain_hint=DOMAIN)
    rows = netops.load_rows(("paraphrase",))[: a.n + 1]
    qs = normalize_questions(netops.router_questions())
    gq = normalize_questions(netops.gate_questions())[0]
    out = {}
    for q in qs + [gq]:
        wall, pn, pms = [], [], []
        states = [netops.gate_state(r["input"], "disable_port", {"port": "1/3"}) if q.qid == "matches_request" else r["input"] for r in rows]
        be.readout_rotation(states[0], q, 0)                        # warm the prefix cache
        for s in states[1:]:
            m = be.readout_rotation(s, q, 0)
            wall.append(m["latency_s"] * 1000); pn.append(m.get("prompt_n") or 0); pms.append(m.get("prompt_ms") or 0)
        out[q.qid] = {"n": len(wall), "wall_ms_median": statistics.median(wall), "wall_ms_p95": sorted(wall)[int(0.95 * (len(wall) - 1))],
                      "prompt_tokens_evaluated_median": statistics.median(pn), "prompt_ms_median": statistics.median(pms),
                      "prompt_tokens_total": m.get("prompt_tokens")}
        print(f"{q.qid:18s} wall median {out[q.qid]['wall_ms_median']:.0f} ms (p95 {out[q.qid]['wall_ms_p95']:.0f}); "
              f"evaluated {out[q.qid]['prompt_tokens_evaluated_median']:.0f} of {m.get('prompt_tokens')} prompt tokens; prompt_ms {out[q.qid]['prompt_ms_median']:.0f}")
    try:
        d = json.load(open(a.out))
    except Exception:
        d = {}
    d[a.tag] = out
    json.dump(d, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
