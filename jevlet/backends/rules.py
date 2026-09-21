"""Deterministic floor: answers built from regexes and lookup tables.

Returns one-hot (or abstain-uniform) distributions. It exists so an application can run the
same decision code with no model at all, and so a model backend has a fallback on timeout.
"""

from __future__ import annotations

import re
from typing import Callable

from .base import Backend
from ..schema import Question, Answer


class RulesBackend(Backend):
    name = "rules"

    def __init__(self, rules: dict[str, Callable[[str, Question], int | None]] | None = None):
        # rules: qid -> function(state, question) -> option index or None (abstain)
        self.rules = rules or {}

    def _decide(self, state: str, questions: list[Question]) -> list[Answer]:
        out = []
        for q in questions:
            fn = self.rules.get(q.qid)
            idx = fn(state, q) if fn else None
            k = q.k
            if idx is None:
                probs = [1.0 / k] * k
            else:
                probs = [0.0] * k
                probs[idx] = 1.0
            out.append(Answer(q.qid, q.kind, probs, None, {"abstained": idx is None}))
        return out


# --- reusable rule helpers for the network-management domain -----------------------
PORT_RE = re.compile(r"\b(?:port|interface|int|gi|g)\s*(?:1/)?(\d{1,2})\b", re.I)
MULTI_RE = re.compile(r"\b(all|every|each|fleet|switches)\b", re.I)


def names_port(state: str) -> bool:
    return PORT_RE.search(state) is not None


def multi_device(state: str) -> bool:
    return MULTI_RE.search(state) is not None
