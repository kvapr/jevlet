"""Calibration and selective-prediction metrics (pure numpy).

All functions take probabilities aligned with a label index. Conventions:
  probs: (N, K) array, rows sum to 1
  y:     (N,) int labels in [0, K)
"""

from __future__ import annotations

import numpy as np


def accuracy(probs, y):
    probs, y = np.asarray(probs), np.asarray(y)
    return float((probs.argmax(1) == y).mean()) if len(y) else float("nan")


def nll(probs, y, eps=1e-12):
    probs, y = np.asarray(probs), np.asarray(y)
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], eps, 1)).mean())


def brier(probs, y):
    """Multiclass Brier score (sum over classes of squared error, averaged over rows)."""
    probs, y = np.asarray(probs, dtype=float), np.asarray(y)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def ece(conf, correct, bins=10, adaptive=False):
    """Expected calibration error on the top-1 confidence.
    conf: (N,) max-probability; correct: (N,) bool. Equal-width bins by default; adaptive = equal-mass."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    n = len(conf)
    if n == 0:
        return float("nan")
    if adaptive:
        order = np.argsort(conf)
        parts = np.array_split(order, bins)
    else:
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
        parts = [np.where(idx == b)[0] for b in range(bins)]
    e = 0.0
    for p in parts:
        if len(p) == 0:
            continue
        e += len(p) / n * abs(correct[p].mean() - conf[p].mean())
    return float(e)


def reliability_table(conf, correct, bins=10):
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                     "conf": float(conf[m].mean()) if m.any() else None,
                     "acc": float(correct[m].mean()) if m.any() else None})
    return rows


def jev_confidence(probs):
    probs = np.asarray(probs, dtype=float)
    k = probs.shape[1]
    if k < 2:
        return np.ones(len(probs))
    return np.clip((k * probs.max(1) - 1.0) / (k - 1.0), 0, 1)


def selective(conf, correct, thresholds=(0.5, 0.7, 0.9, 0.95)):
    """Coverage and error above each confidence threshold (the act/confirm/human buckets)."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=bool)
    out = []
    for t in thresholds:
        m = conf >= t
        out.append({"tau": t, "coverage": float(m.mean()) if len(m) else float("nan"),
                    "n": int(m.sum()),
                    "error_rate": float((~correct[m]).mean()) if m.any() else None,
                    "wrong": int((~correct[m]).sum())})
    return out


def aurc(conf, correct):
    """Area under the risk-coverage curve (lower is better)."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    risks = np.cumsum(err) / (np.arange(len(err)) + 1)
    return float(risks.mean())


def auroc(scores, labels):
    """Rank-based AUROC (ties count half). scores higher => predicts label 1."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    rpos = ranks[: len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_ci(fn, *arrays, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(*[np.asarray(a)[idx] for a in arrays]))
    vals = np.asarray(vals, dtype=float)
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def summarize(probs, y, bins=10):
    probs, y = np.asarray(probs, dtype=float), np.asarray(y)
    pred = probs.argmax(1)
    correct = pred == y
    conf = probs.max(1)
    return {
        "n": int(len(y)),
        "accuracy": float(correct.mean()),
        "nll": nll(probs, y),
        "brier": brier(probs, y),
        "ece": ece(conf, correct, bins),
        "ece_adaptive": ece(conf, correct, bins, adaptive=True),
        "aurc": aurc(conf, correct),
        "mean_conf": float(conf.mean()),
        "selective_jev": selective(jev_confidence(probs), correct),
        "selective_maxp": selective(conf, correct),
    }
