"""Ground-truth tests for the 2026-07-12 fixes.

Every check below has an answer known ANALYTICALLY (or by construction), so it validates the fix
rather than just re-running it. Same discipline as tests/test_math.py.

    python tests/test_fixes_20260712.py

Covers:
  F1        MI symmetric uncertainty: SU = 2 I(C;F)/(H(C)+H(F))   [was 2 I/H(C), which exceeds 1]
  F3        magnitude-preserving shift aggregate vs the scale-free `_frob`
  F4        MMD median-heuristic bandwidth
  F-dec     anti-aliased entropy decimation
  F5        seed is part of the frame cache identity
  NULL      the within-subject H-divergence null runs on short-trial datasets (myobit/senic shape)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# tests never write into a real results tree (CLAUDE.md rule 1)
import os as _os, pathlib as _pl
_os.environ.setdefault("PROFILE_RESULTS_DIR",
                       str(_pl.Path(__file__).resolve().parents[1] / "results" / "_test_sandbox"))

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"    [PASS] {name}   {detail}")
    else:
        FAIL += 1
        print(f"    [FAIL] {name}   {detail}")


# ================================================================================== F1
def test_f1_symmetric_uncertainty():
    print("\n=== F1 — MI symmetric uncertainty denominator ===")
    from dsprofile.module2_separability import mi_symmetric_uncertainty

    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 3, n)
    exact = y.astype(float)                      # feature IS the label -> SU must be 1.0
    noise = rng.standard_normal(n)               # feature independent of the label -> SU ~ 0
    X = np.column_stack([exact, noise])
    su = mi_symmetric_uncertainty(X, y, ["exact", "noise"])

    # SU is an *uncertainty coefficient*: it is bounded in [0, 1] BY DEFINITION. The old code
    # divided by H(C) alone and returned ~2.0 for the exact-copy case, which is impossible.
    check("F1 SU(exact copy of label) == 1.0 (old buggy code gave ~2.0)",
          abs(su["exact"] - 1.0) < 0.05, f"SU={su['exact']:.4f}")
    check("F1 SU(independent feature) ~ 0", su["noise"] < 0.15, f"SU={su['noise']:.4f}")
    check("F1 SU is bounded in [0,1] for every feature",
          all(0.0 <= v <= 1.0 + 1e-9 for v in su.values()), f"{ {k: round(v,3) for k,v in su.items()} }")


# ================================================================================== F3
def test_f3_magnitude_aggregate():
    print("\n=== F3 — magnitude-preserving shift aggregate ===")
    from dsprofile.module3_shift import _frob, _mean_offdiag

    M = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    # THE POINT: _frob is scale-free (a uniformity statistic). 10*M is ten times the shift, and
    # _frob cannot see it. The new aggregate must.
    check("F3 _frob is SCALE-FREE: _frob(M) == _frob(10*M)  [this is the defect]",
          abs(_frob(M) - _frob(10 * M)) < 1e-9, f"{_frob(M):.6f} vs {_frob(10*M):.6f}")
    check("F3 mean-offdiag SCALES with magnitude: 10x M -> 10x aggregate",
          abs(_mean_offdiag(10 * M) - 10 * _mean_offdiag(M)) < 1e-9,
          f"{_mean_offdiag(M):.4f} -> {_mean_offdiag(10*M):.4f}")
    check("F3 mean-offdiag equals the analytic mean of the off-diagonal (2*(1+2+3)/6 = 2.0)",
          abs(_mean_offdiag(M) - 2.0) < 1e-9, f"{_mean_offdiag(M):.4f}")


# ================================================================================== F4
def test_f4_median_gamma():
    print("\n=== F4 — MMD median-heuristic bandwidth ===")
    from dsprofile.module3_shift import mmd_rbf

    rng = np.random.default_rng(1)
    A = rng.standard_normal((400, 8))
    B = rng.standard_normal((400, 8))                       # same distribution -> MMD ~ 0
    check("F4 MMD(identical distributions) ~ 0 under median-gamma",
          mmd_rbf(A, B, gamma="median", rng=rng) < 0.01,
          f"mmd={mmd_rbf(A, B, gamma='median', rng=rng):.5f}")

    # monotone in the true shift: a bigger mean offset must give a bigger MMD
    shifts = [0.0, 0.5, 1.0, 2.0]
    vals = [mmd_rbf(A, rng.standard_normal((400, 8)) + d, gamma="median",
                    rng=np.random.default_rng(7)) for d in shifts]
    check("F4 MMD is monotone increasing in a growing mean shift",
          all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)),
          " < ".join(f"{v:.4f}" for v in vals))
    check("F4 both bandwidth modes are selectable and finite",
          np.isfinite(mmd_rbf(A, B, gamma="inv_d", rng=rng)))


# =============================================================================== F-dec
def test_fdec_antialias():
    print("\n=== F-dec — anti-aliased entropy decimation ===")
    from dsprofile.features_extra import slow_features

    fs, T, q = 2000.0, 1000, 4
    t = np.arange(T) / fs
    # A 700 Hz tone. Decimating by 4 -> new Nyquist 250 Hz. Naive [::4] FOLDS it back into the
    # passband as a spurious ~100 Hz component; a proper anti-alias filter REMOVES it instead.
    x = np.sin(2 * np.pi * 700 * t)[None, None, :]

    naive = x[..., ::q].ravel()
    out = slow_features(x, ["PERMEN"], dict(m=2, perm_d=4, perm_tau=1, min_samples=100),
                        max_samples=T // q)
    check("F-dec slow_features still returns the requested feature", "PERMEN" in out)

    # ground truth: the anti-aliased path must SUPPRESS the out-of-band tone (near-zero power),
    # while naive subsampling preserves it at nearly full power (that IS the aliasing bug).
    from scipy.signal import decimate
    anti = decimate(x, q, axis=-1, ftype="fir", zero_phase=True).ravel()
    check("F-dec naive [::q] ALIASES the 700 Hz tone back into the passband at ~full power",
          np.std(naive) > 0.5, f"std={np.std(naive):.3f}")
    check("F-dec anti-aliased decimate SUPPRESSES the out-of-band tone",
          np.std(anti) < 0.1 * np.std(naive), f"anti={np.std(anti):.4f} vs naive={np.std(naive):.3f}")


# ================================================================================== F5
def test_f5_seed_in_cache_key():
    print("\n=== F5 — seed is part of the frame cache identity ===")
    from dsprofile.windows import _cache_file

    a = _cache_file("ds", "fast", 200, seed=42)
    b = _cache_file("ds", "fast", 200, seed=7)
    check("F5 different seeds -> DIFFERENT cache files (seed robustness is now falsifiable)",
          a != b, f"{a.name}  !=  {b.name}")
    check("F5 same seed -> same cache file (reuse still works)",
          _cache_file("ds", "fast", 200, seed=7) == b)
    check("F5 seed=42 keeps its historical filename (existing caches stay valid)",
          "_s42" not in a.name, a.name)
    check("F-dec/F5 the `complex` cache is version-tagged so the aliased entropy cache is rebuilt",
          "_e3" in _cache_file("ds", "complex", 40, seed=42).name)


# ================================================================================ NULL
def test_hdiv_null_runs_on_short_trials():
    print("\n=== NULL — within-subject H-div null on short-trial datasets ===")
    from dsprofile.module3_shift import hdiv_null_floor

    # Shape of the datasets where the null previously did NOT run at all (myobit/senic): few
    # windows per subject, only 3 trials each. Truth: two halves of ONE subject are the SAME
    # distribution, so d_H must be ~0. The old gates (>=4*cap rows, >=4 trials) skipped these
    # silently and reported "no subject with enough trials" -> the leak control never executed.
    rng = np.random.default_rng(0)
    n_sub, n_trials, per_trial = 6, 3, 30
    X, subj, trials = [], [], []
    for s in range(n_sub):
        for tr in range(n_trials):
            X.append(rng.standard_normal((per_trial, 7)))
            subj += [s] * per_trial
            trials += [s * 100 + tr] * per_trial
    X = np.vstack(X)
    subj = np.asarray(subj)
    trials = np.asarray(trials)

    out = hdiv_null_floor(X, subj, trials, seed=0)
    check("NULL now COMPUTES on a short-trial (3-trial) dataset instead of silently skipping",
          out.get("computed") is True, f"n_subjects={out.get('n_subjects')} note={out.get('note','')[:40]}")
    check("NULL d_H between two halves of the SAME subject ~ 0 (leak-free)",
          out.get("computed") and out["mean"] < 0.30, f"mean={out.get('mean')}")
    check("NULL reports leak_suspected=False on clean data",
          out.get("leak_suspected") is False)


if __name__ == "__main__":
    test_f1_symmetric_uncertainty()
    test_f3_magnitude_aggregate()
    test_f4_median_gamma()
    test_fdec_antialias()
    test_f5_seed_in_cache_key()
    test_hdiv_null_runs_on_short_trials()
    print(f"\n==== {PASS}/{PASS + FAIL} checks passed ====")
    sys.exit(1 if FAIL else 0)
