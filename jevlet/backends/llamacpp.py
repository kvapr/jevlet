"""Next-token probability readout over a llama.cpp server (`llama-server`).

One request per question: the question (instructions + option legend + answer format) goes
in the SYSTEM message so consecutive states share a cached prefix; the state goes in the USER
message; `max_tokens=1, logprobs=true` returns the pre-sampling distribution at the first
generated position. Labels are single tokens in the Qwen vocabulary (A..O, 0..9, Yes/No), so
one position carries the whole answer. Mass on non-label tokens is reported as `label_mass`
(<1 means the model wanted to say something else — treat as a format failure, not a low
probability).

Order bias is real (measured 15.8% prediction flips between two orderings on a 15-way
question); `rotations>1` evaluates cyclic shifts of the option list and averages in option
space, which is the cheapest de-biasing that needs no training.

Requires: llama-server built from llama.cpp with the OpenAI-compatible endpoint (any backend:
CPU, CUDA, Hexagon HTP). GenieX's compiled path does NOT expose logprobs; this backend cannot
target it.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request

from .base import Backend
from ..schema import Question, Answer

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"


def _post(url, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


class LlamaCppReadout(Backend):
    name = "llamacpp-readout"

    def __init__(self, base_url="http://127.0.0.1:18999", rotations=1, top_logprobs=40,
                 domain_hint="", timeout=300):
        self.base_url = base_url.rstrip("/")
        self.rotations = max(1, rotations)
        self.top = top_logprobs
        self.domain_hint = domain_hint  # one sentence of framing prepended to every system prompt
        self.timeout = timeout

    # ---- prompt construction -------------------------------------------------
    def _labels(self, q: Question, order: list[int]) -> tuple[list[str], str]:
        if q.kind == "noul":
            return ["No", "Yes"], "Answer with Yes or No only."
        if q.kind == "score":
            labs = [DIGITS[i] for i in range(len(order))]
            return labs, "Answer with the single digit of the level only."
        labs = [LETTERS[i] for i in range(len(order))]
        return labs, "Answer with the single letter of the option only."

    def _system(self, q: Question, order: list[int]) -> str:
        labs, fmt = self._labels(q, order)
        lines = [self.domain_hint.strip(), q.instructions.strip()]
        if q.kind == "noul":
            lines += q.legend_lines()
        elif q.kind == "score":
            lines.append("Levels (low to high):")
            for lab, i in zip(labs, order):
                lines.append(f"{lab}: {q.options[i]}")
        else:
            lines.append("Options:")
            for lab, i in zip(labs, order):
                d = q.descriptions.get(q.options[i])
                lines.append(f"{lab}. {q.options[i]}" + (f" — {d}" if d else ""))
        lines.append(fmt)
        return "\n".join(l for l in lines if l)

    # ---- one readout ------------------------------------------------------------
    def _readout(self, system: str, state: str, labels: list[str]):
        body = {"model": "x", "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": state}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": self.top,
                "cache_prompt": True,
                # thinking-mode templates (Qwen3.5, Qwen3 non-2507) would otherwise emit "<think>" as the
                # first token and the label readout would see zero label mass
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.perf_counter()
        r = _post(self.base_url + "/v1/chat/completions", body, self.timeout)
        dt = time.perf_counter() - t0
        ch = r["choices"][0]
        first = ch["message"].get("content") or ""
        top = ch["logprobs"]["content"][0]["top_logprobs"]
        mass = [0.0] * len(labels)
        canon = [l.lower() for l in labels]
        for t in top:
            s = t["token"].strip().lower()
            if s in canon:
                mass[canon.index(s)] += math.exp(t["logprob"])
        total = sum(mass)
        usage = r.get("usage", {})
        timings = r.get("timings", {})
        return mass, total, first, dt, usage, timings

    def readout_rotation(self, state: str, q: Question, r: int) -> dict:
        """One rotation of one question: returns probs in CANONICAL option order plus diagnostics.
        Callers that batch many states should loop rotation-major (all states for rotation r, then
        r+1) so the server's prefix cache is reused; `_decide` below is the per-call convenience."""
        k = q.k
        order = list(range(k)) if q.kind == "noul" else [(i + r) % k for i in range(k)]
        labels, _ = self._labels(q, order)
        system = self._system(q, order)
        mass, total, first, dt, usage, timings = self._readout(system, state, labels)
        probs_pos = [m / total if total > 0 else 1.0 / k for m in mass]
        probs = [0.0] * k
        for pos, opt_idx in enumerate(order):
            probs[opt_idx] = probs_pos[pos]
        return {"rotation": r, "probs": probs, "label_mass": total, "first_token": first, "latency_s": dt,
                "prompt_tokens": usage.get("prompt_tokens"), "prompt_ms": timings.get("prompt_ms"),
                "prompt_n": timings.get("prompt_n")}

    @staticmethod
    def combine_rotations(rots: list[dict]) -> tuple[list[float], list[float]]:
        k = len(rots[0]["probs"])
        acc = [sum(r["probs"][i] for r in rots) / len(rots) for i in range(k)]
        logs = [sum(math.log(max(r["probs"][i], 1e-9)) for r in rots) / len(rots) for i in range(k)]
        return acc, logs

    def _decide(self, state: str, questions: list[Question]) -> list[Answer]:
        out = []
        for q in questions:
            rots = 1 if q.kind == "noul" else min(self.rotations, q.k)
            metas = [self.readout_rotation(state, q, r) for r in range(rots)]
            acc, logs = self.combine_rotations(metas)
            out.append(Answer(q.qid, q.kind, acc, logs,
                              {"rotations": metas, "label_mass": sum(m["label_mass"] for m in metas) / rots,
                               "latency_s": sum(m["latency_s"] for m in metas)}))
        return out
