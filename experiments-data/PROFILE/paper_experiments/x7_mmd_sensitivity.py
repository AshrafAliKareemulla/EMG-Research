"""X7 — MMD kernel & aggregation sensitivity. Buries the fixed-gamma / _frob limitations (§9-5, F3/F4).

Recomputes the difficulty correlation (MMD-to-pool vs LOSO) under three bandwidths: median-heuristic,
legacy 1/d, and a multi-kernel gamma grid. Also contrasts the scale-free `_frob` aggregate with the
magnitude-preserving mean off-diagonal. If the sign/conclusion is stable across bandwidths, the result
is not a gamma artifact.

GROUND TRUTH:
  identical distributions -> MMD ~ 0 for median-gamma and EVERY grid kernel;
  increasing mean shift    -> multi-kernel MMD monotone increasing;
  single-kernel biased MMD -> matches the CLOSED-FORM MMD^2 for two isotropic Gaussians.
"""
from __future__ import annotations

import math

import numpy as np

from . import common
from .code_fixes import frob_uniformity, mean_offdiagonal

GAMMA_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


def _difficulty_r(frame, gamma, seed, kernels=None):
    X, _ = common.basis(frame)
    y, subj = frame.label.to_numpy(), frame.subject.to_numpy()
    acc = common.loso_accuracy(X, y, subj, "lda")
    rng = np.random.default_rng(seed)
    mmd = {}
    for s in np.unique(subj):
        this, rest = X[subj == s], X[subj != s]
        if len(this) < 20 or len(rest) < 20:
            continue
        mmd[int(s)] = common.mmd_rbf(this, rest, gamma=gamma, kernels=kernels, rng=rng)
    return common.corr_across_subjects(mmd, acc)


def run_one(dataset, seed=42, n_jobs=1):
    with common.timer(f"X7 :: {dataset}"):
        frame = common.build_frame(dataset, seed=seed)
        res = dict(
            dataset=dataset,
            difficulty_r_median=_difficulty_r(frame, "median", seed),
            difficulty_r_inv_d=_difficulty_r(frame, "1/d", seed),
            difficulty_r_multikernel=_difficulty_r(frame, None, seed, kernels=GAMMA_GRID),
        )
    rs = [res[k]["r"] for k in ("difficulty_r_median", "difficulty_r_inv_d", "difficulty_r_multikernel")]
    rs = [r for r in rs if r == r]
    res["sign_stable_negative"] = bool(rs and all(r < 0 for r in rs))
    res["r_range"] = [float(min(rs)), float(max(rs))] if rs else [float("nan")] * 2
    return res


# ------------------------------------------------ ground truth
def closed_form_mmd2_gaussians(delta_norm2, sigma2, gamma, d):
    """MMD^2 between N(a, sigma2 I) and N(b, sigma2 I) with RBF kernel exp(-gamma||.||^2), ||a-b||^2=delta_norm2."""
    c = 1.0 + 4.0 * gamma * sigma2
    epp = c ** (-d / 2.0)
    epq = c ** (-d / 2.0) * math.exp(-gamma * delta_norm2 / c)
    return float(2.0 * epp - 2.0 * epq)


def selftest(check):
    rng = np.random.default_rng(0)
    d = 4
    A = rng.standard_normal((3000, d))
    B = rng.standard_normal((3000, d))
    # identical distributions -> ~0 for median gamma and every grid kernel
    m_med = common.mmd_rbf(A, B, gamma="median", rng=rng)
    grid = [common.mmd_rbf(A, B, gamma=g, rng=rng) for g in GAMMA_GRID]
    check("X7 identical dists: median-gamma MMD ~ 0", m_med < 0.02, f"{m_med:.4f}")
    check("X7 identical dists: MMD ~ 0 for every grid kernel", max(grid) < 0.03, f"max={max(grid):.4f}")

    # monotonic increase with mean shift (multi-kernel)
    shifts = [0.0, 1.0, 2.0, 4.0]
    mk = [common.mmd_rbf(A, rng.standard_normal((3000, d)) + s, kernels=GAMMA_GRID, rng=rng) for s in shifts]
    check("X7 multi-kernel MMD monotone in mean shift", all(mk[i] < mk[i + 1] for i in range(len(mk) - 1)),
          f"{[round(v,3) for v in mk]}")

    # empirical single-kernel biased MMD vs closed form (isotropic Gaussians)
    g, sig2, shift = 0.5, 1.0, 2.0
    C = rng.standard_normal((4000, d)) + shift
    emp = common.mmd_rbf(A, C, gamma=g, n=4000, rng=rng)
    cf = closed_form_mmd2_gaussians(shift ** 2 * d, sig2, g, d)
    check("X7 empirical MMD matches closed-form Gaussian MMD^2", abs(emp - cf) < 0.03,
          f"emp={emp:.4f} closed={cf:.4f}")

    # F3 aggregate contrast
    M = np.abs(rng.standard_normal((5, 5))); M = (M + M.T) / 2; np.fill_diagonal(M, 0)
    check("X7 _frob scale-free vs mean-offdiag magnitude",
          abs(frob_uniformity(M) - frob_uniformity(9 * M)) < 1e-9
          and abs(mean_offdiagonal(9 * M) - 9 * mean_offdiagonal(M)) < 1e-6)
