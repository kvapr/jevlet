"""Post-hoc calibration: temperature / vector scaling with grouped cross-validation.

Why grouped: paraphrases of one seed request must never be split across fit and eval folds
(a row-wise split leaks the seed's label into the fold it is evaluated on).
"""

from __future__ import annotations

import hashlib
import numpy as np


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fit_temperature(logits, y, grid=None):
    """Scalar temperature minimizing NLL on (logits, y). Grid search (robust for small N)."""
    logits, y = np.asarray(logits, dtype=float), np.asarray(y)
    if grid is None:
        grid = np.exp(np.linspace(np.log(0.05), np.log(50.0), 200))
    best, bestT = np.inf, 1.0
    idx = np.arange(len(y))
    for T in grid:
        p = softmax(logits / T)
        v = -np.log(np.clip(p[idx, y], 1e-12, 1)).mean()
        if v < best:
            best, bestT = v, float(T)
    return bestT


def fit_vector_scaling(logits, y, iters=300, lr=0.05, l2=1e-3):
    """Per-class scale+bias (Guo et al.'s vector scaling) by gradient descent on NLL.
    Use only when N >> K; falls back to temperature otherwise."""
    logits, y = np.asarray(logits, dtype=float), np.asarray(y)
    n, k = logits.shape
    if n < 8 * k:
        T = fit_temperature(logits, y)
        return np.full(k, 1.0 / T), np.zeros(k)
    w, b = np.ones(k), np.zeros(k)
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1
    for _ in range(iters):
        p = softmax(logits * w + b)
        g = p - onehot
        gw = (g * logits).mean(0) + l2 * (w - 1)
        gb = g.mean(0) + l2 * b
        w -= lr * gw
        b -= lr * gb
    return w, b


def group_folds(groups, n_folds=5, seed=0):
    """Deterministic grouped folds: rows with the same group id land in the same fold."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    # shuffle the groups, then deal them round-robin into folds
    fold_of_group = {uniq[i]: int(pos % n_folds) for pos, i in enumerate(perm)}
    return np.array([fold_of_group[g] for g in groups])


def stable_group_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def cv_temperature(logits, y, groups, n_folds=5, seed=0):
    """Out-of-fold temperature-scaled probabilities. Returns (probs_oof, temperatures)."""
    logits, y = np.asarray(logits, dtype=float), np.asarray(y)
    folds = group_folds(groups, n_folds, seed)
    out = np.zeros_like(logits)
    temps = []
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0:
            continue
        T = fit_temperature(logits[tr], y[tr]) if tr.sum() else 1.0
        temps.append(T)
        out[te] = softmax(logits[te] / T)
    return out, temps


def cv_vector(logits, y, groups, n_folds=5, seed=0):
    logits, y = np.asarray(logits, dtype=float), np.asarray(y)
    folds = group_folds(groups, n_folds, seed)
    out = np.zeros_like(logits)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0:
            continue
        w, b = fit_vector_scaling(logits[tr], y[tr])
        out[te] = softmax(logits[te] * w + b)
    return out


def average_permutations(prob_rows):
    """Average distributions obtained under different option orderings (already mapped back
    to canonical option order). prob_rows: list of (K,) arrays -> (K,) array."""
    return np.mean(np.asarray(prob_rows, dtype=float), axis=0)
