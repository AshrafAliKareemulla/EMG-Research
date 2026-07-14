"""Non-parametric stats helpers (Friedman + Wilcoxon signed-rank + Bonferroni).

Accuracy/separability values are non-normal, so we avoid ANOVA (summary 14).
"""
from __future__ import annotations

import itertools

import numpy as np


def friedman(*groups):
    from scipy.stats import friedmanchisquare
    try:
        stat, p = friedmanchisquare(*groups)
        return float(stat), float(p)
    except Exception as e:  # e.g. too few groups / identical values
        return float("nan"), float("nan")


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR. Returns (rejected, q_values) aligned with the input order.

    Phase 1/2 reported dozens of per-dataset and per-predictor correlations uncorrected
    (Module 5: 4 predictors x 14 datasets = 56 tests; meta: 14 predictors). POST_RESULTS_PLAN
    Stage 1 asked for FDR; nothing implemented it. This does.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], bool), np.array([])
    ok = np.isfinite(p)
    q = np.full(n, np.nan)
    rej = np.zeros(n, bool)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return rej, q
    order = idx[np.argsort(p[idx])]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]      # enforce monotonicity
    q[order] = np.minimum(ranked, 1.0)
    rej[order] = q[order] <= alpha
    return rej, q


def fdr_dict(pmap: dict, alpha=0.05):
    """{name: p} -> {name: {"p": p, "q": q, "significant_fdr": bool}}."""
    names = list(pmap)
    rej, q = fdr_bh([pmap[n] for n in names], alpha)
    return {n: dict(p=float(pmap[n]) if pmap[n] == pmap[n] else float("nan"),
                    q=float(qq) if qq == qq else float("nan"),
                    significant_fdr=bool(r))
            for n, qq, r in zip(names, q, rej)}


def wilcoxon_bonferroni(groups: dict):
    """groups: {name: array}. Pairwise Wilcoxon signed-rank with Bonferroni correction."""
    from scipy.stats import wilcoxon
    names = list(groups)
    pairs = list(itertools.combinations(names, 2))
    m = max(1, len(pairs))
    out = []
    for a, b in pairs:
        try:
            _, p = wilcoxon(groups[a], groups[b])
        except Exception:
            p = float("nan")
        out.append(dict(a=a, b=b, p=p, p_bonf=min(1.0, p * m) if p == p else float("nan")))
    return out
