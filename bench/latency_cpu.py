"""CPU latency of the encoder decision models at fixed thread counts (an aarch64 host as a
stand-in for the board's Kryo cores; run the same script on the board for real numbers).

    python -m bench.latency_cpu --models $WORK/models --repo $WORK/verdict-open-jev --threads 1,4 --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from jevlet.schema import normalize_questions
from bench import netops


def time_backend(be, states, qs, n):
    lat = []
    for s in states[:n]:
        t0 = time.perf_counter()
        be._decide(s, qs)
        lat.append(time.perf_counter() - t0)
    return {"n": len(lat), "median_ms": statistics.median(lat) * 1000, "p95_ms": sorted(lat)[int(0.95 * (len(lat) - 1))] * 1000,
            "mean_ms": statistics.mean(lat) * 1000}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--pkg", default=None)
    ap.add_argument("--bundle", default=None, help="a fine-tuned bundle to time as well")
    ap.add_argument("--threads", default="1,4")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/latency-cpu.json")
    a = ap.parse_args()
    rows = netops.load_rows(("canonical", "paraphrase"))
    states = [r["input"] for r in rows]
    qs_all = normalize_questions(netops.router_questions())
    qs_one = [q for q in qs_all if q.qid == "is_write"]
    gate_q = normalize_questions(netops.gate_questions())
    gate_state = netops.gate_state(states[0], "disable_port", {"port": "1/3"})
    out = {"host": os.uname().machine, "cpu": os.cpu_count(), "results": {}}
    for th in [int(t) for t in a.threads.split(",")]:
        import torch
        torch.set_num_threads(th)
        os.environ["OMP_NUM_THREADS"] = str(th)
        from jevlet.backends.encoders import make_backend, OpenJevBackend
        specs = ["verdict", "verdict-torch", "openjev-deberta", "laya", "nli"]
        for spec in specs:
            try:
                be = make_backend(spec, a.models, a.repo, "cpu")
                if hasattr(be, "engine") and be.engine.ort_session is not None:
                    import onnxruntime as ort
                    so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1
                    be.engine.ort_session = ort.InferenceSession(be.engine.ort_session._model_path, so, providers=["CPUExecutionProvider"])
                r = {"router_4q": time_backend(be, states, qs_all, a.n), "noul_1q": time_backend(be, states, qs_one, a.n),
                     "gate_1q": time_backend(be, [gate_state] * a.n, gate_q, a.n)}
                out["results"][f"{be.name}@{th}t"] = r
                print(f"{be.name} @{th} threads: 4q {r['router_4q']['median_ms']:.0f} ms, noul {r['noul_1q']['median_ms']:.0f} ms, gate {r['gate_1q']['median_ms']:.0f} ms", flush=True)
            except Exception as e:
                print("skip", spec, repr(e)[:200])
        if a.bundle:
            be = OpenJevBackend(a.bundle, "cpu", pkg_dir=a.pkg)
            r = {"router_4q": time_backend(be, states, qs_all, a.n), "noul_1q": time_backend(be, states, qs_one, a.n),
                 "gate_1q": time_backend(be, [gate_state] * a.n, gate_q, a.n)}
            out["results"][f"finetuned:{os.path.basename(os.path.dirname(a.bundle))}@{th}t"] = r
            print(f"finetuned bundle @{th} threads: 4q {r['router_4q']['median_ms']:.0f} ms, noul {r['noul_1q']['median_ms']:.0f} ms", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
