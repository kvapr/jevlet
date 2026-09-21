"""Backend protocol. A backend answers a batch of typed questions about one state.

Every backend must return, per question, a probability vector aligned with Question.options
AND the pre-softmax scores it derived it from (log-probs or model logits), so the caller can
re-calibrate. Backends differ in what the probability *means*:

  typesafe   RLCD-trained, vendor-calibrated (cloud)
  encoder    trained decision model (open weights); calibration as shipped, re-fit on your data
  llamacpp   next-token readout of an instruction-tuned LM: over-confident and order-biased
             until temperature-scaled and permutation-averaged
  rules      deterministic label, degenerate one-hot probability (no epistemics at all)

Applications should threshold only after measuring calibration on their own labeled data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..schema import Question, Answer, normalize_questions, state_to_text


@dataclass
class DecisionResult:
    answers: dict[str, Answer]
    latency_s: float
    backend: str
    meta: dict = field(default_factory=dict)

    def to_wire(self, questions: list[Question]) -> dict:
        qmap = {q.qid: q for q in questions}
        return {"model": self.backend,
                "answers": {qid: a.to_wire(qmap[qid]) for qid, a in self.answers.items()},
                "usage": self.meta.get("usage", {})}


class Backend:
    name = "base"

    def decide(self, state: Any, questions: dict[str, dict] | list[Question]) -> DecisionResult:
        qs = questions if isinstance(questions, list) else normalize_questions(questions)
        t0 = time.perf_counter()
        answers = self._decide(state_to_text(state), qs)
        return DecisionResult({a.qid: a for a in answers}, time.perf_counter() - t0, self.name)

    def _decide(self, state: str, questions: list[Question]) -> list[Answer]:
        raise NotImplementedError
