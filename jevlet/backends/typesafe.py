"""Cloud Jev (TypeSafe) client — the reference System One model. Cloud-only; not usable offline.

POST https://api.typesafe.ai/v1/systemone  {state, model, questions}
Returns probabilities per question; we surface log(probabilities) as `logits` so the same
calibration code applies (a no-op if the vendor's calibration holds on your data).
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request

from .base import Backend
from ..schema import Question, Answer


class TypeSafeBackend(Backend):
    name = "typesafe-jev"

    def __init__(self, api_key: str | None = None, model="jev-latest",
                 url="https://api.typesafe.ai/v1/systemone", timeout=30):
        self.key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.key:
            raise RuntimeError("TYPESAFE_API_KEY not set")
        self.model, self.url, self.timeout = model, url, timeout

    def _wire_questions(self, questions: list[Question]) -> dict:
        out = {}
        for q in questions:
            if q.kind == "noul":
                d = {"type": "noul", "instructions": q.instructions}
                if q.criteria_true or q.criteria_false:
                    d["criteria"] = {"true": q.criteria_true, "false": q.criteria_false}
            elif q.kind == "choice":
                d = {"type": "choice", "instructions": q.instructions,
                     "criteria": {o: (q.descriptions.get(o) or None) for o in q.options}}
            else:
                d = {"type": "score", "instructions": q.instructions, "criteria": list(q.options)}
            out[q.qid] = d
        return out

    def _decide(self, state: str, questions: list[Question]) -> list[Answer]:
        body = {"state": state, "model": self.model, "questions": self._wire_questions(questions)}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        t0 = time.perf_counter()
        r = json.load(urllib.request.urlopen(req, timeout=self.timeout))
        dt = time.perf_counter() - t0
        out = []
        for q in questions:
            a = r["answers"][q.qid]
            if q.kind == "noul":
                p = float(a["noul"])
                probs = [1 - p, p]
            elif q.kind == "choice":
                probs = [float(a["probabilities"][o]) for o in q.options]
            else:
                probs = [float(a["probabilities"][str(i)]) for i in range(q.k)]
            logs = [math.log(max(v, 1e-9)) for v in probs]
            out.append(Answer(q.qid, q.kind, probs, logs, {"latency_s": dt, "usage": r.get("usage")}))
        return out
