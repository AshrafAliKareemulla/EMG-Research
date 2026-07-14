"""Regression tests for the 2026-07-10 correctness fixes.

Each test pins one defect found in the results audit, so a future refactor cannot silently
reintroduce it. These are ground-truth tests: synthetic data where the right answer is known.

Run:  python tests/test_corrections.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tests never write into a real results tree (CLAUDE.md rule 1)
import os as _os, pathlib as _pl
_os.environ.setdefault("PROFILE_RESULTS_DIR",
                       str(_pl.Path(__file__).resolve().parents[1] / "results" / "_test_sandbox"))

import numpy as np
import pandas as pd

from dsprofile import config, cv, stats, block_c, block_c as bc, features_extra as fx
from dsprofile.module3_shift import gaussian_kl_split
from dsprofile.windows import _antialias_decimate

R = []


def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}   {detail}")


# ---------------------------------------------------------------------------------------
# 1. The mean/cov split is affine-invariant at ridge=0, and the old ridge breaks it.
# ---------------------------------------------------------------------------------------
def test_affine_invariance():
    rng = np.random.default_rng(0)
    d, n = 12, 400
    A = rng.normal(size=(n, d)) @ np.diag(rng.uniform(.5, 2, d))
    B = rng.normal(size=(n, d)) @ np.diag(rng.uniform(.5, 2, d)) + 1.5
    scale = 10.0 ** rng.uniform(-3, 3, d)                 # raw sEMG-like scale spread
    Ar, Br = A * scale, B * scale
    P = np.vstack([Ar, Br]); mu, sd = P.mean(0), P.std(0)
    Ag, Bg = (Ar - mu) / sd, (Br - mu) / sd

    _, m0, c0 = gaussian_kl_split(Ar, Br, ridge=0.0)
    _, m1, c1 = gaussian_kl_split(Ag, Bg, ridge=0.0)
    rel = max(abs(m0 - m1) / abs(m0), abs(c0 - c1) / abs(c0))
    check("KL split is affine-invariant at ridge=0", rel < 1e-6, f"rel={rel:.2e}")

    _, mr, cr = gaussian_kl_split(Ar, Br, ridge=1e-3)
    _, mg, cg = gaussian_kl_split(Ag, Bg, ridge=1e-3)
    rel_ridge = max(abs(mr - mg) / abs(mr), abs(cr - cg) / abs(cr))
    check("the old ridge=1e-3 DESTROYS invariance (this was the bug)",
          rel_ridge > 1e3 * rel, f"rel_ridge={rel_ridge:.2e} vs rel0={rel:.2e}")

    check("both KL terms are non-negative", m0 >= 0 and c0 >= 0, f"m={m0:.3f} c={c0:.3f}")


# ---------------------------------------------------------------------------------------
# 2. E3 recovers a KNOWN mean/cov ground truth, and refuses to answer when there is no shift.
# ---------------------------------------------------------------------------------------
def _synth_e3(kind, n_ch=3, n_subj=6, nw=1000, seed=1):
    cols = [f"{fb}_c{c}" for fb in config.REPR_BASIS for c in range(n_ch)]
    D = len(cols)
    rng = np.random.default_rng(seed)
    A0 = rng.normal(size=(D, D)) / np.sqrt(D)
    rows, subj = [], []
    for s in range(n_subj):
        if kind == "mean":
            mu, A = rng.normal(0, 2.0, D), A0
        elif kind == "cov":
            mu, A = np.zeros(D), A0 @ np.diag(rng.uniform(.4, 2.5, D))
        else:
            mu, A = np.zeros(D), A0
        rows.append(rng.normal(size=(nw, D)) @ A.T + mu)
        subj += [s] * nw
    X = np.vstack(rows) * (10.0 ** rng.uniform(-3, 3, D))
    f = pd.DataFrame(X, columns=cols)
    f["subject"] = subj; f["session"] = 0; f["repetition"] = 0
    f["label"] = np.tile(np.arange(4), len(f) // 4 + 1)[:len(f)]
    return f


def _e3(frame):
    import dsprofile.windows as W
    orig = W.build_fast_frame
    W.build_fast_frame = lambda ds, seed=42, **k: frame
    try:
        return block_c.meancov_decomposition("synthetic", seed=42)
    finally:
        W.build_fast_frame = orig


def test_e3_ground_truth():
    r = _e3(_synth_e3("mean"))
    p = r["representations"]["pooled"]
    check("E3: mean-dominated -> mean_share ~ 1", p["mean_share_of_excess"] > 0.95,
          f"share={p['mean_share_of_excess']:.4f}")
    check("E3: mean-dominated -> per-subject centering removes ~all divergence",
          r["kl_excess_removed_by_subject_center"] > 0.95,
          f"removed={r['kl_excess_removed_by_subject_center']:.4f}")

    r = _e3(_synth_e3("cov"))
    p = r["representations"]["pooled"]
    check("E3: cov-dominated -> mean_share ~ 0", p["mean_share_of_excess"] < 0.05,
          f"share={p['mean_share_of_excess']:.4f}")
    check("E3: cov-dominated -> centering removes ~nothing",
          abs(r["kl_excess_removed_by_subject_center"]) < 0.05,
          f"removed={r['kl_excess_removed_by_subject_center']:.4f}")

    r = _e3(_synth_e3("null"))
    p = r["representations"]["pooled"]
    check("E3: identical distributions -> shift NOT declared detectable",
          p["shift_detectable"] is False, f"snr={p['snr_excess_over_null']:.3f}")
    check("E3: identical distributions -> mean_share is NaN, not a number",
          not np.isfinite(p["mean_share_of_excess"]))
    check("E3: WITHOUT null correction the null case would falsely read 'cov dominates'",
          p["uncorrected_mean_share"] < 0.2, f"uncorrected={p['uncorrected_mean_share']:.3f}")
    check("E3: no bogus 'removed' fraction when nothing is detectable",
          r["kl_excess_removed_by_subject_zscore"] is None)


# ---------------------------------------------------------------------------------------
# 3. Trial-grouped CV removes the overlapping-window leak.
# ---------------------------------------------------------------------------------------
def test_cv_leak():
    rng = np.random.default_rng(0)
    F, W = 8, 10
    rows = []
    for s in range(6):
        for lab in range(4):
            for rep in range(5):
                trial = rng.normal(0, 3.0, F)                 # strong trial identity
                cls = np.zeros(F); cls[lab % F] = 0.35        # weak class signal
                for _ in range(W):                            # near-duplicate windows
                    rows.append(dict(subject=s, session=0, label=lab, repetition=rep,
                                     **{f"f{i}": v for i, v in
                                        enumerate(trial + cls + rng.normal(0, .05, F))}))
    fr = pd.DataFrame(rows)
    X = fr[[f"f{i}" for i in range(F)]].to_numpy()
    y = fr["label"].to_numpy(); subj = fr["subject"].to_numpy()
    g = cv.trial_ids(fr)
    check("trial_ids: one id per (subject,session,label,repetition)",
          len(np.unique(g)) == 6 * 4 * 5, f"{len(np.unique(g))} ids")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    idx = np.random.default_rng(0).permutation(len(X))
    old = cross_val_score(KNeighborsClassifier(5), X[idx], y[idx], cv=5).mean()
    new = cv.knn_trial_cv(X, y, g, seed=0, max_n=None)
    loso = cv.knn_loso(X, y, subj, seed=0, max_n=None)
    check("old shuffled 5-fold is massively inflated", old > new + 0.25,
          f"old={old:.3f} trial_cv={new:.3f}")
    check("trial-grouped accuracy is near the true weak-signal level", new < 0.5, f"{new:.3f}")
    check("cross-subject <= within-subject", loso <= new + 0.05, f"loso={loso:.3f}")

    # Subsampling thins rows within every trial rather than dropping whole trials. What must
    # hold is (a) every trial survives, and (b) GroupKFold still yields trial-disjoint folds.
    # (Whole-trial selection was the earlier design; it silently collapsed `knn_loso` to one
    # subject when the grouping variable was `subject`.)
    sel = cv._subsample_by_group(g, y, 200, 0)
    check("subsampling keeps every trial represented",
          len(np.unique(g[sel])) == len(np.unique(g)),
          f"{len(np.unique(g[sel]))}/{len(np.unique(g))} trials")
    from sklearn.model_selection import GroupKFold
    gs = g[sel]
    disjoint = all(not (set(gs[tr]) & set(gs[te]))
                   for tr, te in GroupKFold(5).split(np.zeros((len(gs), 1)),
                                                     np.zeros(len(gs)), gs))
    check("subsampled rows still yield trial-disjoint folds", disjoint)


# ---------------------------------------------------------------------------------------
# 4. Anti-aliased decimation.
# ---------------------------------------------------------------------------------------
def test_antialias():
    fs, T = 2000, 500
    t = np.arange(T) / fs
    sig = np.sin(2 * np.pi * 400 * t)[None, None, :]        # 400 Hz: above 500 Hz-Nyquist
    naive = sig[:, :, ::4]
    clean = _antialias_decimate(sig, 4)

    def peak_hz(x, fsx):
        P = np.abs(np.fft.rfft(x.ravel())) ** 2
        return np.fft.rfftfreq(len(x.ravel()), 1 / fsx)[np.argmax(P)]

    check("naive [::4] aliases 400 Hz down to 100 Hz", abs(peak_hz(naive, 500) - 100) < 5)
    check("anti-aliased decimation removes the out-of-band tone",
          (clean ** 2).sum() < 0.05 * (naive ** 2).sum(),
          f"energy ratio {(clean**2).sum()/(naive**2).sum():.2e}")
    check("anti-aliased output has the right length", clean.shape[-1] == T // 4,
          f"{clean.shape[-1]}")

    # in-band content must SURVIVE
    sig2 = np.sin(2 * np.pi * 50 * t)[None, None, :]
    c2 = _antialias_decimate(sig2, 4)
    check("in-band 50 Hz tone survives decimation", abs(peak_hz(c2, 500) - 50) < 5)


# ---------------------------------------------------------------------------------------
# 5. Entropy min-samples guard.
# ---------------------------------------------------------------------------------------
def test_entropy_guard():
    ent = dict(config.ENT)
    names = list(config.SLOW_COMPLEX)
    short = np.random.default_rng(0).normal(size=(2, 1, 50))     # 250 ms @ 200 Hz
    long_ = np.random.default_rng(0).normal(size=(2, 1, 500))    # 250 ms @ 2 kHz
    s = fx.slow_features(short, names, ent)
    l = fx.slow_features(long_, names, ent)
    check("sub-threshold windows -> all entropy features NaN",
          all(np.all(np.isnan(v)) for v in s.values()), f"{list(s)}")
    check("adequate windows -> entropy features finite",
          all(np.isfinite(v).all() for v in l.values()))
    check("ENTROPY_MIN_FS matches min_samples / window",
          abs(config.ENTROPY_MIN_FS - ent["min_samples"] / (config.WINDOW_MS / 1000)) < 1e-9,
          f"{config.ENTROPY_MIN_FS} Hz")


# ---------------------------------------------------------------------------------------
# 6. FDR.
# ---------------------------------------------------------------------------------------
def test_fdr():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222,
         0.43, 0.51, 0.81]
    rej, q = stats.fdr_bh(p, 0.05)
    m = len(p)
    order = np.argsort(p)
    ref = np.minimum.accumulate((np.asarray(p)[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    exp = np.empty(m); exp[order] = np.minimum(ref, 1.0)
    check("BH q-values match the textbook definition", np.allclose(q, exp))
    check("BH q is monotone in p", np.all(np.diff(q[order]) >= -1e-12))
    check("BH rejects exactly q<=alpha", np.array_equal(rej, q <= 0.05), f"{rej.sum()} rejected")
    check("BH is never more liberal than Bonferroni at rank 1",
          q[order][0] >= min(1.0, p[order[0]] * m) - 1e-12)
    rej2, q2 = stats.fdr_bh([np.nan, 0.001, 0.9], 0.05)
    check("BH tolerates NaN p-values", np.isnan(q2[0]) and rej2[1] and not rej2[2])


# ---------------------------------------------------------------------------------------
# 7. Cohort map: the 14 datasets are not 14 independent samples.
# ---------------------------------------------------------------------------------------
def test_cohorts():
    n_cohort = len(set(config.COHORTS.values()))
    check("cohorts < datasets (k=14 overstates independence)", n_cohort < 14, f"{n_cohort} cohorts")
    check("grabmyo_flow_static and _dynamic share a cohort",
          config.COHORTS["grabmyo_flow_static"] == config.COHORTS["grabmyo_flow_dynamic"])
    check("the four EMAHA sets share a cohort",
          len({config.COHORTS[f"emaha_db{i}"] for i in (1, 4, 5, 7)}) == 1)
    check("primary predictor is fixed a priori", config.PRIMARY_PREDICTOR == "mmd_to_pool")


# ---------------------------------------------------------------------------------------
# 8. h_divergence must not memorise trial identity (found by adversarial review).
# ---------------------------------------------------------------------------------------
def _trial_subject(seed, mu=0.0, n_trials=80, per=10, d=8):
    r = np.random.default_rng(seed)
    X, t = [], []
    for k in range(n_trials):
        base = r.normal(0, 3.0, d) + mu           # trial identity, subject-agnostic
        X.append(base + r.normal(0, .05, (per, d)))
        t += [k + seed * 10_000] * per
    return np.vstack(X), np.array(t)


def test_hdiv_leak():
    from dsprofile.module3_shift import h_divergence
    A, ta = _trial_subject(1)
    B, tb = _trial_subject(2)                      # IDENTICAL generating process -> true d_H = 0
    leaked = h_divergence(A, B)                    # no groups -> old behaviour
    honest = h_divergence(A, B, groups_a=ta, groups_b=tb)
    check("h_divergence WITHOUT trial groups saturates on identical distributions",
          leaked > 1.5, f"d_H={leaked:.3f} (max 2.0, truth 0)")

    # The leak's signature is SATURATION near the maximum of 2, so the assertion is a relative
    # collapse, not an absolute cutoff. The honest value is not exactly 0: the two "identical"
    # subjects draw their own finite set of trial centres, so their realised samples differ a
    # little and an RF picks that up. That residue is d_H's finite-trial estimation floor (the
    # analogue of E3's null floor) and it shrinks as the trial count grows. Its exact size is
    # sklearn-version dependent (0.00 here, 0.27 on another box) — hence no tight threshold.
    check("h_divergence WITH trial groups collapses far below the saturation ceiling",
          honest < 0.5 and honest < leaked / 3.0,
          f"leaked={leaked:.3f} honest={honest:.3f} (collapse {leaked/max(honest,1e-9):.1f}x)")

    C, tc = _trial_subject(3, mu=1.5)              # a real shift
    real = h_divergence(A, C, groups_a=ta, groups_b=tc)
    check("h_divergence still detects a real shift", real > honest + 0.1, f"d_H={real:.3f}")
    check("h_divergence is clamped at 0 (a divergence is never negative)", honest >= 0.0)

    # every caller must supply trial ids
    import inspect
    from dsprofile import module5_difficulty as m5
    src = inspect.getsource(m5.subject_shift_stats)
    check("subject_shift_stats accepts trial ids", "trials" in inspect.signature(m5.subject_shift_stats).parameters)


# ---------------------------------------------------------------------------------------
# 9. Subsampling must keep every subject (found by adversarial review).
# ---------------------------------------------------------------------------------------
def test_subsample_keeps_all_groups():
    for nsub, per in [(25, 3600), (40, 9000), (31, 1400)]:
        subj = np.repeat(np.arange(nsub), per)
        sel = cv._subsample_by_group(subj, None, 4000, 0)
        kept = len(np.unique(subj[sel]))
        check(f"knn_subject_cv keeps all {nsub} subjects (was collapsing to 1-2)",
              kept == nsub, f"kept {kept}, rows {len(sel)}")

    # class balance preserved
    g = np.repeat(np.arange(60), 100)
    y = np.repeat(np.arange(6), 1000)
    sel = cv._subsample_by_group(g, y, 1200, 0)
    counts = np.bincount(y[sel])
    check("row subsampling preserves class balance", counts.std() < 1e-9, f"{counts}")

    # and still no trial spans a fold
    from sklearn.model_selection import GroupKFold
    gs = g[sel]
    ok = all(not (set(gs[tr]) & set(gs[te]))
             for tr, te in GroupKFold(5).split(np.zeros((len(gs), 1)), np.zeros(len(gs)), gs))
    check("subsampled rows still yield trial-disjoint folds", ok)
    check("knn_loso is an alias for the honestly-named knn_subject_cv",
          cv.knn_loso is cv.knn_subject_cv)


# ---------------------------------------------------------------------------------------
# 10. The E3 null floor must be TRIAL-disjoint (found by adversarial review).
# ---------------------------------------------------------------------------------------
def _trial_frame(kind, n_sub=6, n_trial=40, per=30, n_ch=3, seed=1):
    cols = [f"{fb}_c{c}" for fb in config.REPR_BASIS for c in range(n_ch)]
    D = len(cols)
    r = np.random.default_rng(seed)
    A0 = r.normal(size=(D, D)) / np.sqrt(D)
    rows = []
    for s in range(n_sub):
        if kind == "mean":
            mu, A = r.normal(0, 2.0, D), A0
        elif kind == "cov":
            mu, A = np.zeros(D), A0 @ np.diag(r.uniform(.4, 2.5, D))
        else:
            mu, A = np.zeros(D), A0
        for t in range(n_trial):
            base = r.normal(size=D) @ A.T + mu                  # trial identity
            win = base + r.normal(0, .08, (per, D)) @ A.T * 0.15  # near-duplicate windows
            for w in range(per):
                rows.append(dict(subject=s, session=0, label=t % 4, repetition=t,
                                 **{c: v for c, v in zip(cols, win[w])}))
    f = pd.DataFrame(rows)
    for c in cols:
        f[c] = f[c] * (10.0 ** r.uniform(-3, 3))
    return f


def test_null_floor_trial_disjoint():
    from dsprofile.module3_shift import gaussian_kl_split
    f = _trial_frame("null")
    subj = f.subject.to_numpy(); trials = cv.trial_ids(f)
    X = f[[c for c in f.columns if "_c" in c]].to_numpy(float)
    rng = np.random.default_rng(0)

    # a subject's rows, split by ROW (old) vs by TRIAL (new)
    idx = np.where(subj == 0)[0]
    N = 300
    ridx = rng.permutation(idx)[:2 * N]
    _, m, c = gaussian_kl_split(X[ridx[:N]], X[ridx[N:]], ridge=0.0)
    null_row = m + c
    halves = bc._matched_halves(subj, trials, 20, N, np.random.default_rng(0))
    h1, h2 = halves[0]
    _, m, c = gaussian_kl_split(X[h1], X[h2], ridge=0.0)
    null_trial = m + c
    check("row-permuted null UNDER-estimates the floor (trial-mates shared across halves)",
          null_row < 0.5 * null_trial, f"row={null_row:.2f} trial-disjoint={null_trial:.2f}")

    # halves must be trial-disjoint by construction
    disjoint = all(not (set(trials[a]) & set(trials[b])) for a, b in halves.values())
    check("matched halves are trial-disjoint", disjoint)
    same_n = all(len(a) == len(b) == N for a, b in halves.values())
    check("matched halves have identical row counts", same_n)

    # end-to-end: no true shift -> not detectable
    import dsprofile.windows as W
    orig = W.build_fast_frame
    W.build_fast_frame = lambda ds, seed=42, **k: f
    try:
        r = bc.meancov_decomposition("synth", seed=42)
    finally:
        W.build_fast_frame = orig
    p = r["representations"]["pooled"]
    check("trial-structured null data -> shift NOT detectable", p["shift_detectable"] is False,
          f"snr={p['snr_excess_over_null']:.3f}")
    check("null floor is recorded as trial-disjoint", p.get("null_is_trial_disjoint") is True)

    # and a real mean shift IS still recovered under trial structure
    f2 = _trial_frame("mean")
    W.build_fast_frame = lambda ds, seed=42, **k: f2
    try:
        r2 = bc.meancov_decomposition("synth", seed=42)
    finally:
        W.build_fast_frame = orig
    p2 = r2["representations"]["pooled"]
    check("trial-structured mean shift -> detected, mean-dominated",
          p2["shift_detectable"] and p2["mean_share_of_excess"] > 0.9,
          f"share={p2['mean_share_of_excess']:.3f}")


if __name__ == "__main__":
    for t in (test_affine_invariance, test_e3_ground_truth, test_cv_leak, test_antialias,
              test_entropy_guard, test_fdr, test_cohorts,
              test_hdiv_leak, test_subsample_keeps_all_groups, test_null_floor_trial_disjoint):
        print(f"\n--- {t.__name__} ---")
        t()
    print(f"\n==== {sum(R)}/{len(R)} checks passed ====")
    sys.exit(0 if all(R) else 1)
