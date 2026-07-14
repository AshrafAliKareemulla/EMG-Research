"""Code fixes F1-F4 + F-dec, each a small pure function with a GROUND-TRUTH test.

These bury the low-level correctness limitations flagged in EXTERNAL_RESEARCH_REVIEW.md §4/§9-7:
  F1  MI symmetric-uncertainty denominator  (was 2*I/H(C); correct is 2*I/(H(C)+H(F)))
  F2  FuzzyEn amplitude invariance          (exp(-(d/tol)^n) vs the non-invariant exp(-(d^n)/tol))
  F3  magnitude-preserving shift aggregate  (mean off-diagonal beside the scale-free _frob)
  F-dec anti-aliased decimation             (scipy.signal.decimate vs naive [::q])
F4 (MMD median-gamma) and F5 (seed in cache key) live in common.mmd_rbf / windows respectively.
"""
from __future__ import annotations

import numpy as np

from . import common
from .common import EPS, safe_div


# ============================== F1 — symmetric uncertainty ============================
def _entropy(a):
    _, c = np.unique(np.asarray(a), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p + EPS)).sum())


def _joint_entropy(a, b):
    ab = np.stack([np.asarray(a), np.asarray(b)], axis=1)
    _, c = np.unique(ab, axis=0, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p + EPS)).sum())


def _quantile_bin(x, bins):
    x = np.asarray(x, float)
    if np.unique(x).size <= bins:
        return np.unique(x, return_inverse=True)[1]
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    return np.digitize(x, edges)


def symmetric_uncertainty(feature, y, bins=10):
    """SU = 2*I(C;F) / (H(C) + H(F)), F quantile-binned. In [0,1]; 1 = F determines C, 0 = independent.

    This is the CORRECT normalisation. The shipped `module2.mi_symmetric_uncertainty` divides by H(C)
    alone, so its value can exceed 1 and its ranking collapses to raw mutual information.
    """
    f = np.asarray(feature)
    if not np.issubdtype(f.dtype, np.integer):
        f = _quantile_bin(f, bins)
    yb = np.unique(np.asarray(y), return_inverse=True)[1]
    fb = np.unique(f, return_inverse=True)[1]
    hf, hc = _entropy(fb), _entropy(yb)
    mi = hf + hc - _joint_entropy(fb, yb)
    return float(safe_div(2.0 * mi, hf + hc, 0.0))


def buggy_su(feature, y, bins=10):
    """The SHIPPED behaviour (2*I/H(C)) — kept only to demonstrate the defect in the GT test."""
    f = np.asarray(feature)
    if not np.issubdtype(f.dtype, np.integer):
        f = _quantile_bin(f, bins)
    yb = np.unique(np.asarray(y), return_inverse=True)[1]
    fb = np.unique(f, return_inverse=True)[1]
    hf, hc = _entropy(fb), _entropy(yb)
    mi = hf + hc - _joint_entropy(fb, yb)
    return float(safe_div(2.0 * mi, hc, 0.0))


# ============================== F2 — fuzzy entropy invariance =========================
def _cheb_matrix(emb):
    n = emb.shape[0]
    d = np.zeros((n, n))
    for k in range(emb.shape[1]):
        col = emb[:, k]
        d = np.maximum(d, np.abs(col[:, None] - col[None, :]))
    return d


def fuzzy_entropy(x, m=2, n=5, r=0.30, invariant=True):
    """FuzzyEn with a membership toggle.

    invariant=True  -> exp(-(d/tol)^n)  : dimensionally consistent, amplitude-INVARIANT (the fix).
    invariant=False -> exp(-(d^n)/tol)  : the shipped / source-paper form, amplitude-SENSITIVE for n!=1.
    tol = r * std(x). Baseline-removed embeddings, self-matches excluded.
    """
    x = np.asarray(x, float)
    N = len(x)
    if N < m + 2:
        return np.nan
    tol = r * (x.std() + EPS)

    def phi(mm):
        e = np.array([x[i:i + mm] for i in range(N - m)], float)
        if e.shape[0] < 2:
            return np.nan
        e = e - e.mean(axis=1, keepdims=True)
        d = _cheb_matrix(e)
        mu = np.exp(-((d / tol) ** n)) if invariant else np.exp(-(d ** n) / tol)
        np.fill_diagonal(mu, 0.0)
        return mu.sum() / ((N - m) * (N - m - 1))

    pm, pm1 = phi(m), phi(m + 1)
    if not (pm > 0 and pm1 > 0):
        return np.nan
    return float(np.log(pm) - np.log(pm1))


def amplitude_sensitivity(x, scales=(0.1, 10.0), invariant=True, **kw):
    """Max |FuzzyEn(a*x) - FuzzyEn(x)| over scales a — 0 for a truly amplitude-invariant estimator."""
    base = fuzzy_entropy(x, invariant=invariant, **kw)
    return float(max(abs(fuzzy_entropy(a * np.asarray(x, float), invariant=invariant, **kw) - base)
                     for a in scales))


# ============================== F3 — shift-matrix aggregates ==========================
def frob_uniformity(M):
    """The SHIPPED `_frob`: RMS of off / off.max() — SCALE-FREE (a 'uniformity' statistic)."""
    M = np.asarray(M, float)
    off = M[~np.eye(len(M), dtype=bool)]
    if off.size == 0:
        return float("nan")
    scaled = off / (off.max() + EPS)
    return float(np.sqrt((scaled ** 2).sum()) / len(off) ** 0.5)


def mean_offdiagonal(M):
    """Magnitude-PRESERVING aggregate: plain mean of the off-diagonal (scales with the shift)."""
    M = np.asarray(M, float)
    off = M[~np.eye(len(M), dtype=bool)]
    return float(off.mean()) if off.size else float("nan")


# ============================== F-dec — anti-aliased decimation =======================
def antialias_decimate(x, q):
    """Zero-phase FIR decimation by q (removes >Nyquist/q content); naive fallback for very short x."""
    from scipy.signal import decimate
    x = np.asarray(x, float)
    if len(x) <= 27 * q:
        return x[::q]
    return np.asarray(decimate(x, q, ftype="fir", zero_phase=True))


def naive_decimate(x, q):
    return np.asarray(x, float)[::q]


# ============================== ground-truth selftest =================================
def selftest(check):
    rng = np.random.default_rng(0)

    # F1: SU=1 for a perfect (label) predictor, ~high for a noisy copy, 0 for independence,
    # and buggy_su (2*I/H(C)) exceeds 1 on the copy.
    y = rng.integers(0, 4, 600)
    su_copy = symmetric_uncertainty(y, y, bins=4)                       # integer feature == label -> exact 1
    su_noisy = symmetric_uncertainty(y.astype(float) + 0.15 * rng.standard_normal(600), y, bins=8)
    su_indep = symmetric_uncertainty(rng.standard_normal(600), y, bins=10)
    check("F1 SU(label copy) == 1", abs(su_copy - 1.0) < 1e-9, f"{su_copy:.3f}")
    check("F1 SU(noisy copy) is high", su_noisy > 0.4, f"{su_noisy:.3f}")
    check("F1 SU(independent) ~ 0", su_indep < 0.1, f"{su_indep:.3f}")
    check("F1 buggy 2*I/H(C) exceeds 1 on the copy (the defect)",
          buggy_su(y.astype(float), y, bins=4) > 1.5, f"{buggy_su(y.astype(float), y, bins=4):.3f}")

    # F2: invariant FuzzyEn is amplitude-blind; the shipped form is not (n=5)
    sig = np.cumsum(rng.standard_normal(300))
    check("F2 invariant FuzzyEn: FE(x)==FE(10x)", amplitude_sensitivity(sig, invariant=True) < 1e-6,
          f"sens={amplitude_sensitivity(sig, invariant=True):.2e}")
    check("F2 shipped form is amplitude-sensitive at n=5",
          amplitude_sensitivity(sig, invariant=False) > 1e-3,
          f"sens={amplitude_sensitivity(sig, invariant=False):.2e}")

    # F3: _frob is scale-free; mean off-diagonal scales 10x
    M = np.abs(rng.standard_normal((6, 6)))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0)
    check("F3 _frob(M) == _frob(10M) (scale-free)", abs(frob_uniformity(M) - frob_uniformity(10 * M)) < 1e-9)
    check("F3 mean_offdiag(10M) == 10*mean_offdiag(M) (magnitude)",
          abs(mean_offdiagonal(10 * M) - 10 * mean_offdiagonal(M)) < 1e-6)

    # F-dec: an above-(new-Nyquist) tone is attenuated by antialias, aliased (kept) by naive
    fs, q = 1000, 4
    t = np.arange(2048) / fs
    tone = np.sin(2 * np.pi * 400 * t)            # 400 Hz; new Nyquist after /4 = 125 Hz
    e_anti = float(np.mean(antialias_decimate(tone, q) ** 2))
    e_naive = float(np.mean(naive_decimate(tone, q) ** 2))
    check("F-dec anti-alias attenuates the >Nyquist tone", e_anti < 0.05, f"E={e_anti:.3f}")
    check("F-dec naive subsampling aliases it back (energy retained)", e_naive > 0.3, f"E={e_naive:.3f}")
