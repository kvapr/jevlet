"""Question / answer schema, shaped after the public Jev API.

Request (one call):
    state: str | dict | list               the facts; text only
    questions: {id: Question}              evaluated in parallel against the same state

Question kinds:
    noul   {"type":"noul",   "instructions":"Is X true?",  "criteria": {"true": ..., "false": ...} (optional)}
    choice {"type":"choice", "instructions":"Which ...?",  "criteria": {opt: description | {what,not_for,examples} | None}}
    score  {"type":"score",  "instructions":"How ...?",    "criteria": ["level 0 desc", "level 1 desc", ...]}  (2..10, low->high)

Answer shapes:
    noul   {"noul": p_yes}
    choice {"choice": opt, "probabilities": {opt: p}, "confidence": c}
    score  {"score": expected_level, "probabilities": {"0": p0, ...}, "legend": {"0": desc, ...}, "confidence": c}

Confidence for choice/score is the concentration statistic Jev documents:
    c = (K * max_p - 1) / (K - 1)         (1 for a point mass, 0 for uniform)

Every backend in this package also returns the raw log-probabilities it computed, so the
same answer can be re-calibrated (temperature / vector scaling) in application code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

NOUL_LABELS = ("no", "yes")


@dataclass
class Question:
    qid: str
    kind: str                      # "noul" | "choice" | "score"
    instructions: str
    options: list[str]             # noul: ["no","yes"]; choice: option keys; score: level descriptions
    descriptions: dict[str, str] = field(default_factory=dict)  # option -> description text (choice)
    criteria_true: str | None = None    # noul
    criteria_false: str | None = None   # noul

    @property
    def k(self) -> int:
        return len(self.options)

    def legend_lines(self) -> list[str]:
        """Human-readable option list used by prompt-based backends."""
        if self.kind == "noul":
            lines = []
            if self.criteria_true:
                lines.append(f"Yes means: {self.criteria_true}")
            if self.criteria_false:
                lines.append(f"No means: {self.criteria_false}")
            return lines
        if self.kind == "score":
            return [f"{i}: {d}" for i, d in enumerate(self.options)]
        out = []
        for o in self.options:
            d = self.descriptions.get(o)
            out.append(f"{o}: {d}" if d else o)
        return out


def _describe(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        parts = []
        if v.get("what"):
            parts.append(str(v["what"]))
        if v.get("not_for"):
            parts.append(f"Not for: {v['not_for']}")
        if v.get("examples"):
            ex = v["examples"]
            ex = ex if isinstance(ex, list) else [ex]
            parts.append("Examples: " + "; ".join(str(e) for e in ex))
        return " ".join(parts)
    return json.dumps(v)


def normalize_questions(questions: dict[str, dict]) -> list[Question]:
    """TypeSafe-style request map -> list[Question]. Validates cardinalities."""
    out = []
    for qid, q in questions.items():
        kind = q["type"]
        instr = q["instructions"]
        crit = q.get("criteria")
        if kind == "noul":
            ct = cf = None
            if isinstance(crit, dict):
                ct, cf = crit.get("true"), crit.get("false")
            out.append(Question(qid, "noul", instr, list(NOUL_LABELS), criteria_true=ct, criteria_false=cf))
        elif kind == "choice":
            if isinstance(crit, dict):
                opts = list(crit.keys())
                desc = {o: _describe(v) for o, v in crit.items()}
            else:
                opts = [str(o) for o in (crit or [])]
                desc = {}
            if not 2 <= len(opts) <= 255:
                raise ValueError(f"{qid}: choice needs 2..255 options, got {len(opts)}")
            out.append(Question(qid, "choice", instr, opts, desc))
        elif kind == "score":
            levels = [str(x) for x in (crit or [])]
            if not 2 <= len(levels) <= 10:
                raise ValueError(f"{qid}: score needs 2..10 levels, got {len(levels)}")
            out.append(Question(qid, "score", instr, levels))
        else:
            raise ValueError(f"{qid}: unknown type {kind}")
    return out


def confidence(probs: list[float]) -> float:
    k = len(probs)
    if k < 2:
        return 1.0
    return max(0.0, min(1.0, (k * max(probs) - 1.0) / (k - 1.0)))


@dataclass
class Answer:
    qid: str
    kind: str
    probs: list[float]                 # aligned with Question.options
    logits: list[float] | None = None  # log-probabilities (or model logits) before any post-hoc scaling
    meta: dict = field(default_factory=dict)  # backend diagnostics: label_mass, first_token, latency, ...

    def to_wire(self, q: Question) -> dict:
        p = self.probs
        if self.kind == "noul":
            return {"type": "noul", "noul": float(p[1])}
        if self.kind == "choice":
            i = max(range(len(p)), key=lambda j: p[j])
            return {"type": "choice", "choice": q.options[i],
                    "probabilities": {o: float(v) for o, v in zip(q.options, p)},
                    "confidence": confidence(p)}
        # score
        exp = float(sum(i * v for i, v in enumerate(p)))
        return {"type": "score", "score": exp,
                "probabilities": {str(i): float(v) for i, v in enumerate(p)},
                "legend": {str(i): d for i, d in enumerate(q.options)},
                "confidence": confidence(p)}


# Convenience constructors (TypeSafe-shaped dicts)
def Noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    d = {"type": "noul", "instructions": instructions}
    if true or false:
        d["criteria"] = {"true": true, "false": false}
    return d


def Choice(instructions: str, criteria: dict | list) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def Score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def state_to_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {state_to_text(v)}" for k, v in state.items())
    if isinstance(state, list):
        return "\n".join(state_to_text(v) for v in state)
    return json.dumps(state)
