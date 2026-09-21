"""Open encoder decision models (non-generative, one forward pass per state) behind the jevlet
Backend interface. Each wrapper maps our Question objects onto the model's own request shape
and maps its answer back to probabilities aligned with Question.options.

  OpenJevBackend     com-kotobalabs/open-jev-deberta-v3-large (DeBERTa-v3-large, 0.4B) — bundled `typed_decisions`
  LayaBackend        convaiinnovations/laya (ModernBERT-large 421M, RLCD) — `pip install laya`
  VerdictBackend     heman10x/rlcd-modernbert-151m (GLiClass ModernBERT-base 151M, ONNX) — Verdict-open-jev repo
  NLIZeroShotBackend any MNLI cross-encoder (poorjev-style): entailment of "the request asks to <option>"

All of them are English text-in, distribution-out; none of them generates tokens.
"""

from __future__ import annotations

import math
import os
import sys
import time

from .base import Backend
from ..schema import Question, Answer


def _logs(probs):
    return [math.log(max(p, 1e-9)) for p in probs]


def _norm(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


def _opt_text(q: Question, o: str) -> str:
    d = q.descriptions.get(o)
    return f"{o}: {d}" if d else o


def _short(desc: str | None, max_chars: int = 90) -> str | None:
    """First clause of a description, for encoders with a 512-token budget shared by all options."""
    if not desc:
        return None
    d = desc.split(" Not for:")[0].split(" Examples:")[0].split(". ")[0].strip()
    return d[:max_chars]


def _desc(q: Question, o: str, mode: str) -> str:
    if mode == "bare":
        return o.replace("_", " ")
    d = q.descriptions.get(o)
    if mode == "short":
        d = _short(d)
    return d or o.replace("_", " ")


class OpenJevBackend(Backend):
    name = "open-jev-deberta-v3-large"

    def __init__(self, model_dir: str, device: str | None = None, pkg_dir: str | None = None):
        # the published checkpoint bundles `typed_decisions`; a bundle we trained does not, so a
        # package dir (the kotoba-lang/typed-decisions checkout, its src/) can be given instead
        if pkg_dir:
            sys.path.insert(0, pkg_dir)
        else:
            sys.path.insert(0, model_dir)
        from typed_decisions.open_jev import OpenJev
        self.m = OpenJev.from_pretrained(model_dir, device=device)
        self.name = os.path.basename(model_dir.rstrip("/"))

    @staticmethod
    def _req(q: Question, bare=False) -> dict:
        if q.kind == "noul":
            return {"type": "noul", "instructions": q.instructions}
        return {"type": q.kind, "instructions": q.instructions,
                "options": list(q.options) if bare else [_opt_text(q, o) for o in q.options]}

    def _decide(self, state: str, questions: list[Question]) -> list[Answer]:
        # Preferred: ALL questions in one forward pass, exactly as these models are trained (the span
        # head is position-aware: a bundle trained on packed batteries answers a lone question
        # differently — measured 0.55 vs 0.07 on the command question). The published DeBERTa
        # checkpoint's 512-token context cannot hold state + a 24-option question with descriptions,
        # so fall back to one question per pass, then to bare option names.
        reqs = [self._req(q) for q in questions]
        t0 = time.perf_counter()
        mode = "packed"
        try:
            res = self.m.decide(state, reqs)
        except ValueError:
            mode = "per-question"
            res = []
            for q, req in zip(questions, reqs):
                try:
                    res.append(self.m.decide(state, [req])[0])
                except ValueError:
                    req["options"] = list(q.options) if q.kind != "noul" else None
                    res.append(self.m.decide(state, [self._req(q, bare=True)])[0])
        dt = time.perf_counter() - t0
        out = []
        for q, r, req in zip(questions, res, reqs):
            if q.kind == "noul":
                p = float(r["noul"])
                probs = [1 - p, p]
            else:
                pm = r["probabilities"]
                keys = req["options"] if req.get("options") else list(q.options)
                probs = _norm([float(pm[o]) for o in keys])
            out.append(Answer(q.qid, q.kind, probs, _logs(probs), {"latency_s": dt / len(questions), "mode": mode,
                                                                  "confidence_reported": r.get("confidence")}))
        return out


class LayaBackend(Backend):
    name = "laya"

    def __init__(self, model_dir: str, subfolder: str | None = None, device: str | None = None, desc_mode: str = "full"):
        os.environ.setdefault("USE_TF", "0")
        import laya
        self.agent = laya.load(model_dir, device=device, subfolder=subfolder)
        self.desc_mode = desc_mode
        self.name = "laya" + (f"-{subfolder}" if subfolder else "") + (f"-{desc_mode}" if desc_mode != "full" else "")

    def _decide(self, state: str, questions: list[Question]) -> list[Answer]:
        qmap = {}
        for q in questions:
            if q.kind == "noul":
                qmap[q.qid] = {"type": "noul", "instructions": q.instructions}
            elif q.kind == "choice":
                qmap[q.qid] = {"type": "choice", "instructions": q.instructions,
                               "criteria": {o: _desc(q, o, self.desc_mode) for o in q.options}}
            else:
                qmap[q.qid] = {"type": "score", "instructions": q.instructions, "criteria": list(q.options)}
        t0 = time.perf_counter()
        res = self.agent.predict(state, qmap)
        dt = time.perf_counter() - t0
        out = []
        for q in questions:
            a = res["answers"][q.qid]
            if q.kind == "noul":
                p = float(a["noul"])
                probs = [1 - p, p]
            else:
                pm = a.get("probabilities") or {}
                if all(o in pm for o in q.options):
                    probs = [float(pm[o]) for o in q.options]
                elif all(str(i) in pm for i in range(q.k)):
                    probs = [float(pm[str(i)]) for i in range(q.k)]
                else:  # positional fallback
                    vals = list(pm.values())
                    probs = [float(v) for v in vals[: q.k]] + [0.0] * max(0, q.k - len(vals))
                probs = _norm(probs)
            out.append(Answer(q.qid, q.kind, probs, _logs(probs), {"latency_s": dt / len(questions),
                                                                  "confidence_reported": a.get("confidence")}))
        return out


class VerdictBackend(Backend):
    """openJev-verdict-2.0 via the Verdict-open-jev repo (`rlcd` package, which imports `core` from the repo root)."""
    name = "openjev-verdict-151m"

    def __init__(self, model_dir: str, repo_dir: str, device: str = "cpu", use_onnx: bool = True, desc_mode: str = "short"):
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from rlcd import DecisionEngine  # noqa
        # DecisionEngine looks for <model_dir>/model.onnx when device == cpu; remove it from the search to force torch
        self.engine = DecisionEngine(model_name_or_path=model_dir, device=device)
        if not use_onnx:
            self.engine.ort_session = None
        # GLiClass packs ALL option labels before the text in one 512-token window: long catalog
        # descriptions push the state out of the window entirely (measured: 0% on the 24-way question)
        self.desc_mode = desc_mode
        self.name = "openjev-verdict-151m" + ("-onnx" if self.engine.ort_session is not None else "-torch") + (f"-{desc_mode}" if desc_mode != "full" else "")

    def _decide(self, state: str, questions: list[Question]) -> list[Answer]:
        from rlcd import Choice, Option, Score, Level, Noul
        queries = []
        for q in questions:
            if q.kind == "noul":
                prop = q.instructions
                if q.criteria_true:
                    prop += f" (Yes means: {q.criteria_true})"
                queries.append(Noul(id=q.qid, proposition=prop, semantics="conditional_on_sufficient_evidence_v2"))
            elif q.kind == "choice":
                queries.append(Choice(id=q.qid, question=q.instructions,
                                      options=tuple(Option(id=o, description=_desc(q, o, self.desc_mode)) for o in q.options[:24])))
            else:
                queries.append(Score(id=q.qid, question=q.instructions,
                                     levels=tuple(Level(id=str(i), description=d, value=float(i)) for i, d in enumerate(q.options))))
        t0 = time.perf_counter()
        res = self.engine.evaluate(state, queries)
        dt = time.perf_counter() - t0
        out = []
        for q, r in zip(questions, res.results):
            pm = dict(r.probabilities)
            abst = float(pm.get("__insufficient_evidence__", 0.0))
            if q.kind == "noul":
                pt = float(pm.get("true", 0.0)); pf = float(pm.get("false", 0.0))
                probs = _norm([pf, pt])
            elif q.kind == "choice":
                probs = _norm([float(pm.get(o, 0.0)) for o in q.options])
            else:
                probs = _norm([float(pm.get(str(i), 0.0)) for i in range(q.k)])
            out.append(Answer(q.qid, q.kind, probs, _logs(probs),
                              {"latency_s": dt / len(questions), "p_abstain": abst,
                               "is_abstention": bool(getattr(r, "is_abstention", False))}))
        return out


class NLIZeroShotBackend(Backend):
    """Zero-shot via an NLI cross-encoder: P(option) ∝ exp(entailment logit of the option hypothesis)."""
    name = "nli-zeroshot"

    def __init__(self, model_dir: str, device: str | None = None, template="The request asks to {}."):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.dev)
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent = labels.get("entailment", 0)
        self.con = labels.get("contradiction", 2)
        self.template = template
        self.name = "nli-" + os.path.basename(model_dir.rstrip("/"))

    def _decide(self, state: str, questions: list[Question]) -> list[Answer]:
        import torch
        out = []
        for q in questions:
            if q.kind == "noul":
                hyps = [q.instructions.rstrip("?") + "."]
            else:
                hyps = [self.template.format(q.descriptions.get(o) or o) for o in q.options]
            t0 = time.perf_counter()
            enc = self.tok([state] * len(hyps), hyps, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.dev)
            with torch.no_grad():
                logits = self.model(**enc).logits.float().cpu()
            dt = time.perf_counter() - t0
            if q.kind == "noul":
                z = logits[0]
                pe = torch.softmax(torch.stack([z[self.con], z[self.ent]]), 0).tolist()
                probs = pe
            else:
                ent = logits[:, self.ent]
                probs = torch.softmax(ent, 0).tolist()
            out.append(Answer(q.qid, q.kind, probs, _logs(probs), {"latency_s": dt}))
        return out


def make_backend(spec: str, models_root: str, repo_root: str | None = None, device: str | None = None) -> Backend:
    """spec: openjev-deberta | laya[-td][:full|short|bare] | verdict[-torch][:full|short|bare] | nli"""
    m = lambda d: os.path.join(models_root, d)
    base, _, mode = spec.partition(":")
    if base == "openjev-deberta":
        return OpenJevBackend(m("com-kotobalabs_open-jev-deberta-v3-large"), device)
    if base == "laya":
        return LayaBackend(m("convaiinnovations_laya"), None, device, desc_mode=mode or "full")
    if base == "laya-td":
        return LayaBackend(m("convaiinnovations_laya"), "typed-decisions", device, desc_mode=mode or "full")
    if base in ("verdict", "verdict-torch"):
        return VerdictBackend(m("heman10x_rlcd-modernbert-151m"), repo_root, device or "cpu", use_onnx=(base == "verdict"), desc_mode=mode or "short")
    if base == "nli":
        return NLIZeroShotBackend(m("cross-encoder_nli-deberta-v3-base"), device)
    raise ValueError(spec)
