"""Module 3 — distribution shift, quantified (the core novelty).

Pairwise between-subject (and, where sessions exist, between-session) distances on z-scored
representative features:
  * MMD (RBF kernel), energy distance,
  * closed-form Gaussian-KL split into a MEAN term and a COVARIANCE term (summaries 09/11),
  * H-divergence via pairwise subject-classifier error (RandomForest, summary 07).
Aggregate each matrix to a rescaled Frobenius scalar. Answers A4 (inter-subject vs inter-day).
"""
from __future__ import annotations

import json
import itertools

import numpy as np
import pandas as pd

from . import config, cv, windows, progress


def _basis(frame):
    cols = []
    for fb in config.REPR_BASIS:
        cols += [c for c in frame.columns if c.startswith(fb + "_c")]
    X = np.nan_to_num(frame[cols].to_numpy(np.float64))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd


def _sample(X, n, rng):
    return X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]


def median_gamma(A, B, rng, n=200):
    """RBF bandwidth by the median heuristic: gamma = 1/(2 * median ||x_i - x_j||^2).

    F4. The old default (gamma = 1/d, sklearn's) is fixed w.r.t. the data scale, so MMD magnitudes
    are not comparable across datasets of different dimension/spread. The median heuristic is the
    field standard and adapts to the data. `config.MMD_GAMMA` selects which one is PRIMARY; X7
    reports the sensitivity envelope across both plus a multi-kernel sum.
    """
    from scipy.spatial.distance import pdist
    Z = np.vstack([_sample(A, n, rng), _sample(B, n, rng)])
    d2 = pdist(Z, "sqeuclidean")
    med = float(np.median(d2)) if d2.size else 0.0
    return 1.0 / (2.0 * med) if med > 1e-12 else 1.0 / max(1, A.shape[1])


def mmd_rbf(A, B, gamma=None, n=400, rng=None):
    rng = rng or np.random.default_rng(0)
    A, B = _sample(A, n, rng), _sample(B, n, rng)
    if gamma is None:
        gamma = getattr(config, "MMD_GAMMA", "median")
    if gamma == "median":
        gamma = median_gamma(A, B, rng)
    elif gamma in ("inv_d", None):
        gamma = 1.0 / A.shape[1]
    from sklearn.metrics.pairwise import rbf_kernel
    Kaa = rbf_kernel(A, A, gamma).mean()
    Kbb = rbf_kernel(B, B, gamma).mean()
    Kab = rbf_kernel(A, B, gamma).mean()
    return float(max(0.0, Kaa + Kbb - 2 * Kab))


def energy_distance(A, B, n=400, rng=None):
    rng = rng or np.random.default_rng(0)
    A, B = _sample(A, n, rng), _sample(B, n, rng)
    from scipy.spatial.distance import cdist
    dab = cdist(A, B).mean(); daa = cdist(A, A).mean(); dbb = cdist(B, B).mean()
    return float(max(0.0, 2 * dab - daa - dbb))


def gaussian_kl_split(A, B, ridge=1e-3):
    """KL(N_A || N_B) split into its mean and covariance terms (summary 09/11).

        mean_term = 1/2 (mu1-mu0)^T S1^-1 (mu1-mu0)          >= 0
        cov_term  = 1/2 [tr(S1^-1 S0) - d + ln(detS1/detS0)] >= 0   (Stein's loss)
        total     = mean_term + cov_term = KL(N_A || N_B)

    IMPORTANT (2026-07-10 audit) — SCALE INVARIANCE.
    With ``ridge=0`` both terms are EXACTLY invariant under any invertible global affine map
    x -> A x + b applied to both A and B. A global z-score is such a map, so comparing this
    split on "raw" vs "globally z-scored" features measures nothing (verified: 4e-12 relative
    difference). Only a PER-SUBJECT map changes between-subject divergence. See block_c.E3.

    The default ``ridge=1e-3`` is NOT scale-invariant and silently breaks that identity — it
    is retained only for the distance-to-pool predictors in module5/sdi, where the input is
    already globally z-scored so the ridge is on a sensible scale and acts as a mild
    conditioner. Any analysis that *interprets* the mean/cov split must pass ``ridge=0`` and
    ensure n > d (truncate with PCA if necessary).
    """
    mu0, mu1 = A.mean(0), B.mean(0)
    d = A.shape[1]
    S0 = np.cov(A, rowvar=False)
    S1 = np.cov(B, rowvar=False)
    if ridge:
        S0 = S0 + np.eye(d) * ridge
        S1 = S1 + np.eye(d) * ridge
    S1inv = np.linalg.pinv(S1)
    mean_term = float((mu1 - mu0) @ S1inv @ (mu1 - mu0))
    _, logdet1 = np.linalg.slogdet(S1)
    _, logdet0 = np.linalg.slogdet(S0)
    cov_term = float(np.trace(S1inv @ S0) - d + (logdet1 - logdet0))
    total = 0.5 * (mean_term + cov_term)
    return total, 0.5 * mean_term, 0.5 * cov_term


def _sample_idx(n_rows, n, rng):
    return np.arange(n_rows) if n_rows <= n else rng.choice(n_rows, n, replace=False)


def h_divergence(A, B, n=400, rng=None, groups_a=None, groups_b=None):
    """d_H = 2(1-2*err) of a group-vs-group classifier (RF, 5-fold) - summary 07.

    LEAKAGE (fixed 2026-07-10). This previously shuffled the rows (`_sample` -> `rng.choice`,
    which fires whenever len > n, i.e. on the real call path) and then ran a plain
    `cross_val_score(cv=5)`. Windows overlap 50 % and cluster by trial, and trial identity is
    perfectly predictive of which group a window came from, so the RF memorised trial
    fingerprints from the training fold. Measured on two groups drawn from an IDENTICAL
    distribution (true d_H = 0), the old code returned **1.975** against a maximum of 2.0; the
    trial-grouped estimate returns ~0. Phase-1's "H-divergence uniformly high (0.72-0.98) on
    every dataset" was that artifact, not a property of EMG.

    Pass `groups_a` / `groups_b` (trial ids aligned with the rows of A / B) so folds keep each
    trial whole. Without them the estimate is not trustworthy and a warning is emitted rather
    than an inflated number returned silently.
    """
    rng = rng or np.random.default_rng(0)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, GroupKFold

    if groups_a is None or groups_b is None:
        from . import progress
        progress.log("WARNING h_divergence without trial groups -> leaked estimate")
        A2, B2 = _sample(A, n, rng), _sample(B, n, rng)
        X = np.vstack([A2, B2]); y = np.r_[np.zeros(len(A2)), np.ones(len(B2))]
        err = 1 - cross_val_score(RandomForestClassifier(n_estimators=20, random_state=0),
                                  X, y, cv=5).mean()
        return float(2 * (1 - 2 * err))

    ia = _sample_idx(len(A), n, rng)
    ib = _sample_idx(len(B), n, rng)
    X = np.vstack([A[ia], B[ib]])
    y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    ga = np.asarray(groups_a)[ia].astype(np.int64)
    gb = np.asarray(groups_b)[ib].astype(np.int64)
    gb = gb + (int(ga.max()) + 1 if ga.size else 0) + 1      # B's trial ids cannot collide
    g = np.r_[ga, gb]
    n_splits = int(min(5, len(np.unique(g))))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return float("nan")
    err = 1 - cross_val_score(RandomForestClassifier(n_estimators=20, random_state=0),
                              X, y, cv=GroupKFold(n_splits=n_splits), groups=g).mean()
    # d_H is a divergence: clamp estimator noise at 0 rather than report a negative divergence
    # when the two groups are genuinely indistinguishable.
    return float(max(0.0, 2 * (1 - 2 * err)))



def _pair_metrics(GA, GB, seed):
    """All five distances for one subject/session pair, computed in ONE shot (KL covariance
    factorised once, not twice). Runs inside a joblib worker.

    GA/GB are (rows, trial_ids) tuples; the trial ids make d_H leak-free.
    """
    (A, ta), (B, tb) = GA, GB
    rng = np.random.default_rng(seed)
    _, kl_mean, kl_cov = gaussian_kl_split(A, B)
    return (mmd_rbf(A, B, rng=rng), energy_distance(A, B, rng=rng),
            h_divergence(A, B, rng=rng, groups_a=ta, groups_b=tb), kl_mean, kl_cov)


def _pairwise_all(groups, seed, n_jobs, desc):
    """One parallel pass over all pairs -> five symmetric matrices. O(n^2) pairs, but each
    pair is computed once and the loop is spread across all CPU cores (joblib/loky)."""
    from joblib import Parallel, delayed
    keys = list(groups)
    pairs = [(i, j) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    with progress.timer(f"{desc}: {len(pairs)} pairs over {len(keys)} groups"):
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_pair_metrics)(groups[keys[i]], groups[keys[j]], seed + k)
            for k, (i, j) in enumerate(pairs))
    n = len(keys)
    names = ["mmd", "energy", "hdiv", "kl_mean", "kl_cov"]
    M = {nm: np.zeros((n, n)) for nm in names}
    for (i, j), vals in zip(pairs, res):
        for nm, v in zip(names, vals):
            M[nm][i, j] = M[nm][j, i] = v
    return keys, M


def _frob(M):
    """⚠ SCALE-FREE. This is a UNIFORMITY statistic, not a magnitude: it divides the off-diagonal
    by its own max, so `M` and `10*M` score identically and an all-equal matrix scores 1.0 whatever
    its magnitude. Kept for continuity with the committed results — but never quote it as "how much
    shift there is". Use `_mean_offdiag` for that (F3)."""
    off = M[~np.eye(len(M), dtype=bool)]
    if off.size == 0:
        return float("nan")
    scaled = off / (off.max() + 1e-12)
    return float(np.sqrt((scaled ** 2).sum()) / len(off) ** 0.5)


def _mean_offdiag(M):
    """F3 — magnitude-PRESERVING aggregate: the plain mean of the off-diagonal. Unlike `_frob`,
    `_mean_offdiag(10*M) == 10 * _mean_offdiag(M)`, so it can be compared across datasets."""
    off = M[~np.eye(len(M), dtype=bool)]
    return float(np.mean(off)) if off.size else float("nan")


def _grouped(X, keyarr, cap, rng, min_n=20, trials=None):
    """Split X by key, drop groups < min_n, cap each group to `cap` rows.

    Returns {key: (rows, trial_ids)}. The trial ids ride along so `h_divergence` can build
    trial-disjoint folds; without them its RF memorises trial fingerprints and every pairwise
    d_H saturates near 2 (see `h_divergence`).
    """
    out = {}
    for k in sorted(np.unique(keyarr)):
        idx = np.where(keyarr == k)[0]
        if len(idx) < min_n:
            continue
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        t = None if trials is None else np.asarray(trials)[idx]
        out[int(k)] = (X[idx], t)
    return out


def hdiv_null_floor(X, subj, trials, seed=42, cap=400, n_subjects=6):
    """d_H between two TRIAL-DISJOINT halves of the SAME subject. Must be ~0.

    This is the only honest leak diagnostic for H-divergence. A high d_H between two DIFFERENT
    subjects is expected and real (people are separable in feature space). A high d_H between
    two halves of ONE subject can only mean the classifier is memorising trial identity. Under
    the pre-2026-07-10 code this returned ~1.97 against a maximum of 2.0.
    """
    rng = np.random.default_rng(seed)
    # The requirement is only that a subject can be split into two TRIAL-DISJOINT halves with
    # enough rows on each side to fit the pairwise RF. The old gates (>= 4*cap rows AND >= 4
    # trials AND >= 50 rows/half) were absolute, so `myobit` and `senic` — which have fewer
    # windows per subject — produced NO null at all, i.e. the leak control silently did not run
    # on 2/14 datasets. Gates are now relative and the shortfall is REPORTED, never silent.
    MIN_TRIALS, MIN_ROWS_PER_HALF = 2, 25
    order = sorted(np.unique(subj), key=lambda s: -(subj == s).sum())     # biggest subjects first
    vals, skipped = [], []
    for s in order:
        if len(vals) >= n_subjects:
            break
        idx = np.where(subj == s)[0]
        ts = np.unique(trials[idx])
        if len(ts) < MIN_TRIALS:
            skipped.append((int(s), f"{len(ts)} trial(s)"))
            continue
        ts = rng.permutation(ts)
        h = max(1, len(ts) // 2)
        i1 = idx[np.isin(trials[idx], ts[:h])]
        i2 = idx[np.isin(trials[idx], ts[h:])]
        if min(len(i1), len(i2)) < MIN_ROWS_PER_HALF:
            skipped.append((int(s), f"halves {len(i1)}/{len(i2)} rows"))
            continue
        vals.append(h_divergence(X[i1][:cap], X[i2][:cap], rng=rng,
                                 groups_a=trials[i1][:cap], groups_b=trials[i2][:cap]))
    if not vals:
        return dict(computed=False, n_subjects=0, skipped=skipped[:6],
                    note="NULL NOT COMPUTED: no subject could be split into two trial-disjoint "
                         "halves with >=25 rows each. The H-divergence leak control did NOT run "
                         "on this dataset — do not claim it is leak-free.")
    v = np.asarray(vals, float)
    return dict(computed=True, mean=float(v.mean()), max=float(v.max()), n_subjects=len(vals),
                n_skipped=len(skipped), leak_suspected=bool(v.mean() > 0.5),
                note="d_H between trial-disjoint halves of the SAME subject; ~0 if leak-free, "
                     "~1.97 under the pre-fix shuffled-CV code")


def run(dataset, seed=42, n_jobs=None):
    config.ensure_dirs()
    n_jobs = config.resolve_jobs() if n_jobs is None else n_jobs
    frame = windows.build_fast_frame(dataset, seed=seed)
    X = _basis(frame)
    rng = np.random.default_rng(seed)
    cap = 600                                              # per-group window cap for pairwise

    trials = cv.trial_ids(frame)                           # for leak-free h_divergence folds
    subj_groups = _grouped(X, frame.subject.to_numpy(), cap, rng, trials=trials)
    keys, M = _pairwise_all(subj_groups, seed, n_jobs, f"module3 {dataset} inter-subject")

    result = dict(
        dataset=dataset, n_subjects=len(keys),
        hdiv_within_subject_null=hdiv_null_floor(X, frame.subject.to_numpy(), trials, seed),
        inter_subject=dict(
            mmd_frob=_frob(M["mmd"]), energy_frob=_frob(M["energy"]), hdiv_frob=_frob(M["hdiv"]),
            kl_mean_frob=_frob(M["kl_mean"]), kl_cov_frob=_frob(M["kl_cov"]),
            # F3 — magnitude-preserving companions to the scale-free `*_frob` above. QUOTE THESE
            # when the claim is "how much shift", not "how uniform the shift matrix is".
            mmd_mean_offdiag=_mean_offdiag(M["mmd"]),
            energy_mean_offdiag=_mean_offdiag(M["energy"]),
            hdiv_mean_offdiag=_mean_offdiag(M["hdiv"]),
            kl_mean_mean_offdiag=_mean_offdiag(M["kl_mean"]),
            kl_cov_mean_offdiag=_mean_offdiag(M["kl_cov"]),
            mmd_gamma=getattr(config, "MMD_GAMMA", "median"),
            # DEPRECATED — do not report. This ratio is computed with the default ridge=1e-3,
            # which breaks the affine invariance of the mean/cov split, and it is not corrected
            # against the within-subject estimation-noise floor. The publishable decomposition
            # is block_c.E3 -> representations.pooled.mean_share_of_excess.
            mean_vs_cov_ratio_DEPRECATED=float(np.nanmean(M["kl_mean"][M["kl_mean"] > 0]) /
                                               (np.nanmean(M["kl_cov"][M["kl_cov"] > 0]) + 1e-12)),
        ),
        deprecations=["inter_subject.mean_vs_cov_ratio_DEPRECATED -> use block_c E3",
                      "A4_inter_subject_over_inter_day_mmd_DEPRECATED -> use block_c E2"],
    )

    # inter-day (A4) where sessions exist
    if frame.session.nunique() > 1:
        sess_groups = _grouped(X, frame.session.to_numpy(), cap, rng, trials=trials)
        if len(sess_groups) > 1:
            _, SM = _pairwise_all(sess_groups, seed, n_jobs, f"module3 {dataset} inter-day")
            result["inter_day"] = dict(mmd_frob=_frob(SM["mmd"]), hdiv_frob=_frob(SM["hdiv"]),
                                       n_sessions=len(sess_groups))
            # DEPRECATED — granularity mismatch: the numerator is subject-vs-subject while the
            # denominator is session-POOL vs session-POOL (all subjects mixed into each session).
            # Pooling shrinks the between-session distance, so this ratio is biased upward and
            # its Phase-1 values (<1) were an artifact. block_c.E2 compares like with like.
            result["A4_inter_subject_over_inter_day_mmd_DEPRECATED"] = (
                result["inter_subject"]["mmd_frob"] / (result["inter_day"]["mmd_frob"] + 1e-12))

    outdir = config.RESULTS_DIR / "module3"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"{dataset}__shift_matrices.npz", subjects=np.array(keys),
             mmd=M["mmd"], energy=M["energy"], hdiv=M["hdiv"],
             kl_mean=M["kl_mean"], kl_cov=M["kl_cov"])
    (outdir / f"{dataset}__shift.json").write_text(json.dumps(result, indent=2))
    return result
