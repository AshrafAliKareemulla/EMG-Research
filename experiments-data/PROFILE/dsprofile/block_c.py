"""Block C — distribution-shift completions.

E2  A4-fix : inter-subject (cross-subject, within a session) vs inter-day (within-subject,
             cross-session) — a FAIR comparison (Phase-1 conflated the two granularities).
E3  Mean-vs-covariance decomposition of the between-subject Gaussian KL.
E4  Conditional-shift disparity matrix (Albuquerque): pairwise kNN label-disagreement, the
             *conditional* counterpart to Module-3's *marginal* H-divergence.

E3 — WHAT CHANGED AND WHY (2026-07-10 audit)
--------------------------------------------
The original E3 compared the mean/cov split on RAW vs GLOBALLY Z-SCORED features and expected
the mean term to dominate on raw and the covariance term after z-scoring (Yoneda). That
experiment cannot work, because the split is *exactly invariant* to any invertible global
affine map:

    KL mean term  = 1/2 (mu1-mu0)^T S1^-1 (mu1-mu0)
    KL cov  term  = 1/2 [ tr(S1^-1 S0) - d + ln(det S1 / det S0) ]

Under x -> A x + b applied to BOTH subjects:  mu1-mu0 -> A(mu1-mu0),  S -> A S A^T, so
(mu1-mu0)^T A^T (A S1 A^T)^-1 A (mu1-mu0) = (mu1-mu0)^T S1^-1 (mu1-mu0), and
tr((A S1 A^T)^-1 A S0 A^T) = tr(S1^-1 S0), while ln(det S1/det S0) is unchanged (the |det A|^2
factors cancel). A global z-score IS such a map. Verified numerically to 4e-12 relative error.
The differences the old code reported came entirely from the `+1e-3*I` ridge, which is not
scale-invariant and therefore bites completely differently when raw feature scales span 1000x.
Those numbers were regularisation artifacts.

What DOES change the between-subject divergence is a *per-subject* map — a different affine
transform for each subject. So this module now reports three representations:

  * ``pooled``          — global z-score (identical to raw, up to the invariance; asserted)
  * ``subject_center``  — each subject's own mean removed  -> isolates the MEAN contribution
  * ``subject_zscore``  — each subject's own mean and scale removed

The drop in total KL from ``pooled`` -> ``subject_center`` is, by definition, the share of
between-subject divergence attributable to mean shift. That is the quantity Yoneda predicts is
large, and it is measurable.

Why the OLD numbers came out nearly identical (163.4 vs 140.5 on grabmyo) rather than wildly
different: the old contrast changed the SIGNAL-level normalisation, and on the representative
basis a global per-channel signal scale x -> a*x induces almost exactly a diagonal affine map
on the feature vector (measured, not assumed):

    MAV, WL, RMS    scale by a           (homogeneous degree 1)
    HJ_MOB, HJ_COM  unchanged            (degree 0)
    MFL             shifts by log10(a)   (additive constant)
    WAMP            threshold-based      (the ONE genuinely nonlinear feature)

So 6 of 7 basis features move under a map the estimator is provably blind to. The residual
differences the old E3 reported were the ridge, plus a little WAMP. This module therefore works
entirely in FEATURE space, where the algebra is exact, and asserts the invariance numerically.

Two further corrections:
  * ``ridge=0`` with a sample covariance, plus PCA truncation to keep n/d >= 10, instead of a
    scale-dependent ridge. (Ledoit-Wolf is *also* not affine-invariant — it shrinks toward
    tr(S)/d * I — so it cannot be used for the invariance assertion.)
  * a NULL FLOOR from TRIAL-DISJOINT split-halves of the SAME subject. With d~40 and n=400 the
    covariance term carries a large positive estimation bias; without subtracting it,
    "covariance dominates" is an artifact of estimating S from finite samples, not a property
    of EMG. Validated on synthetic data: for two IDENTICAL distributions the uncorrected
    statistic reports a mean share of 0.07, i.e. "covariance dominates", from noise alone.

    The halves MUST be split by trial, not by row. A row-level permutation scatters a trial's
    50 %-overlapping windows across both halves, so the halves share trials and look far more
    alike than two independent samples: on trial-structured data the null came out ~14x too
    small, and the "excess" above it was mostly noise — reintroducing the very artifact the
    null exists to remove. Both halves also carry the same number of independent TRIALS as the
    between-subject groups, because the effective sample size of correlated windows is set by
    the trial count, not the row count.
"""
from __future__ import annotations

import json

import numpy as np

from . import config, cv, windows, progress
from .module3_shift import mmd_rbf, gaussian_kl_split, _basis

# A between-subject shift counts as detectable only if its excess over the within-subject
# estimation-noise floor exceeds this fraction of that floor. Guards against computing
# mean/cov "shares" from a ratio of noise to noise.
DETECT_FRAC = 0.25


# --------------------------------------------------------------------------------------
# E2 — fair inter-subject vs inter-day
# --------------------------------------------------------------------------------------
def _groups(X, keyarr, cap, rng, min_n=20):
    out = {}
    for k in sorted(np.unique(keyarr)):
        g = X[keyarr == k]
        if len(g) < min_n:
            continue
        out[int(k)] = g if len(g) <= cap else g[rng.choice(len(g), cap, replace=False)]
    return out


def _pairwise_mmd(groups, rng):
    keys = list(groups)
    return [mmd_rbf(groups[keys[i]], groups[keys[j]], rng=rng)
            for i in range(len(keys)) for j in range(i + 1, len(keys))]


def a4_fair(frame, dataset, seed=42):
    """E2. Fair inter-subject vs inter-day comparison (needs >1 session).

    Both sides are measured at the SAME granularity (group-vs-group MMD between individual
    subjects / individual sessions). Phase 1 compared a session-POOL (all subjects mixed)
    against subject-vs-subject, which is why its ratios came out < 1.

    STATISTICS (fixed 2026-07-10). The first version ran a Mann-Whitney over the *pairs*:
    2709 inter-subject pairs from 43 grabmyo subjects, 917 from 14 senic subjects. Those pairs
    are massively non-independent -- each subject appears in ~2(n-1) of them -- so the test was
    pseudo-replicated and returned absurd p-values (senic p = 3.9e-108). The effect itself was
    never in doubt (rank-biserial 0.80-0.89); only the p-value was fiction.

    The honest unit of analysis is the SUBJECT. For each subject with >= 2 sessions we form
      inter_day(s)     = mean MMD between that subject's own sessions
      inter_subject(s) = mean MMD between that subject and every other subject, within a session
    and run a paired Wilcoxon signed-rank across subjects (n = #subjects, not #pairs). The
    pair-level means are still reported as descriptive statistics.
    """
    if frame.session.nunique() < 2:
        return dict(applicable=False, note="single-session dataset; A4 not applicable")
    X = _basis(frame); rng = np.random.default_rng(seed)
    subj = frame.subject.to_numpy(); sess = frame.session.to_numpy()
    cap = 400

    # ---- per-subject inter-subject MMD (cross-subject, within a session) -------------------
    per_subj_is = {}
    inter_subj = []
    for ss in np.unique(sess):
        m = sess == ss
        g = _groups(X[m], subj[m], cap, rng)
        keys = list(g)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                v = mmd_rbf(g[keys[i]], g[keys[j]], rng=rng)
                inter_subj.append(v)
                per_subj_is.setdefault(keys[i], []).append(v)
                per_subj_is.setdefault(keys[j], []).append(v)

    # ---- per-subject inter-day MMD (cross-session, within a subject) -----------------------
    per_subj_id = {}
    inter_day = []
    for s in np.unique(subj):
        m = subj == s
        g = _groups(X[m], sess[m], cap, rng)
        vals = _pairwise_mmd(g, rng) if len(g) > 1 else []
        if vals:
            inter_day += vals
            per_subj_id[int(s)] = float(np.mean(vals))

    if not inter_subj or not inter_day:
        return dict(applicable=False,
                    note="not enough subjects/sessions after the >=20-window filter")

    a, b = np.asarray(inter_subj), np.asarray(inter_day)

    # ---- subject-level paired test (the honest one) ---------------------------------------
    common = sorted(set(per_subj_is) & set(per_subj_id))
    paired = None
    if len(common) >= 5:
        from scipy.stats import wilcoxon
        xa = np.array([np.mean(per_subj_is[s]) for s in common])
        xb = np.array([per_subj_id[s] for s in common])
        try:
            w, pw = wilcoxon(xa, xb, alternative="greater")
        except ValueError:
            w, pw = float("nan"), float("nan")
        n_gt = int((xa > xb).sum())
        paired = dict(n_subjects=len(common), wilcoxon_stat=float(w), p_value=float(pw),
                      n_subjects_inter_subject_greater=n_gt,
                      frac_subjects_inter_subject_greater=float(n_gt / len(common)),
                      median_ratio=float(np.median(xa / (xb + 1e-12))),
                      unit_of_analysis="subject (paired)")

    # pair-level Mann-Whitney kept ONLY as a descriptive effect size (rank-biserial)
    from scipy.stats import mannwhitneyu
    try:
        u, p_pairs = mannwhitneyu(a, b, alternative="greater")
        eff = float(2 * u / (len(a) * len(b)) - 1)
    except ValueError:
        u, p_pairs, eff = float("nan"), float("nan"), float("nan")

    caveat = None
    if dataset.startswith("senic"):
        caveat = ("senic 'sessions' are electrode-shift / rotation / fatigue CONDITIONS, not days. "
                  "Its inter-day axis is a shift-robustness axis and must not be pooled with "
                  "grabmyo's true multi-day sessions. Most senic subjects have a single session, "
                  "so this estimate rests on the few with >=2.")
    if dataset.startswith("grabmyo_flow"):
        caveat = ("grabmyo_flow_static and grabmyo_flow_dynamic are the same 20-subject cohort "
                  "(same 12 channels, same 3 sessions); they are not independent evidence for A4.")

    return dict(
        applicable=True,
        inter_subject_within_session_mmd=float(a.mean()),
        inter_subject_mmd_std=float(a.std()), n_inter_subject_pairs=int(len(a)),
        inter_day_within_subject_mmd=float(b.mean()),
        inter_day_mmd_std=float(b.std()), n_inter_day_pairs=int(len(b)),
        n_subjects_with_multiple_sessions=len(per_subj_id),
        ratio_inter_subject_over_inter_day=float(a.mean() / (b.mean() + 1e-12)),
        subject_level_test=paired,
        p_value=(paired or {}).get("p_value", float("nan")),   # the quotable p
        rank_biserial=eff,
        pairwise_mannwhitney_p_PSEUDOREPLICATED=float(p_pairs),
        pairwise_p_warning=("pairs share subjects and are NOT independent; this p-value is "
                            "pseudo-replicated and must not be quoted. Use subject_level_test."),
        interpretation=("inter-subject > inter-day" if a.mean() > b.mean()
                        else "inter-day > inter-subject"),
        caveat=caveat,
    )


# --------------------------------------------------------------------------------------
# E3 — mean vs covariance, done correctly
# --------------------------------------------------------------------------------------
def _basis_cols(frame):
    cols = []
    for fb in config.REPR_BASIS:
        cols += [c for c in frame.columns if c.startswith(fb + "_c")]
    return cols


def _repr_matrix(frame, rep):
    """Basis matrix under one representation.

    `pooled` is a global z-score. `subject_center` / `subject_zscore` apply a DIFFERENT affine
    map per subject — the only kind of map that can change between-subject divergence.
    """
    X = np.nan_to_num(frame[_basis_cols(frame)].to_numpy(np.float64))
    subj = frame.subject.to_numpy()

    if rep == "raw":                       # only the invariance check uses this (no PCA)
        return X

    # The global z-score must come FIRST for EVERY representation. It is a no-op for the KL
    # split itself (which is affine-invariant), but the downstream PCA truncation is NOT
    # invariant: built on raw-scale features, whose scales span ~1e6 here, the top-d components
    # collapse onto the few large-scale features and the covariance in the remaining directions
    # goes numerically singular. With ridge=0 that makes tr(S1^-1 S0) explode.
    #
    # Observed in the 2026-07-10 run BEFORE this fix: `subject_center` cov_term on ninapro_db2
    # was 2.8e9 against a `pooled` cov_term of 3.1e3, driving
    # `kl_excess_removed_by_subject_center` to -597703. `pooled` was safe (already z-scored) and
    # `subject_zscore` was safe (per-feature per-subject scaling is scale-free); only
    # `subject_center` was exposed.
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / np.where(sd < 1e-12, 1.0, sd)     # common global scale for ALL reps
    if rep == "pooled":
        return Z
    if rep in ("subject_center", "subject_zscore"):
        Y = Z.copy()
        for s in np.unique(subj):
            m = subj == s
            mu_s = Y[m].mean(0)
            if rep == "subject_center":
                Y[m] = Y[m] - mu_s
            else:
                sd_s = Y[m].std(0)
                Y[m] = (Y[m] - mu_s) / np.where(sd_s < 1e-12, 1.0, sd_s)
        return Y
    raise ValueError(rep)


def _pca_project(X, n_comp, seed=0):
    """Pooled PCA truncation to condition the covariance estimate.

    A pooled PCA is a global linear map, so it cannot by itself remove between-subject mean
    shift. Truncating to n/d >= 10 is what stops the cov term being dominated by sample-
    covariance estimation error.
    """
    from sklearn.decomposition import PCA
    n_comp = int(min(n_comp, X.shape[1], max(2, X.shape[0] - 1)))
    return PCA(n_components=n_comp, random_state=seed).fit_transform(X), n_comp


def _matched_halves(subj, trials, n_trials_half, n_rows, rng):
    """Split every subject's TRIALS into two disjoint halves of exactly `n_trials_half` trials,
    and draw exactly `n_rows` rows from each half.

    This is the heart of the null-floor correction. An earlier version permuted ROWS, which
    scattered a trial's 50 %-overlapping windows across both halves. The two halves then shared
    trials, so they looked far more alike than two genuinely independent samples — the null was
    under-estimated ~14x on trial-structured data and the "excess" over it was mostly noise,
    reintroducing exactly the artifact the null exists to remove.

    Both halves carry the same number of *independent trials* as well as the same number of
    rows, because both KL terms shrink with the effective sample size, and the effective sample
    size of correlated windows is set by the trial count, not the row count.

    Returns {subject: (rows_half1, rows_half2)}.
    """
    out = {}
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        ts = np.unique(trials[idx])
        if len(ts) < 2 * n_trials_half:
            continue
        ts = rng.permutation(ts)
        halves = []
        for h in (ts[:n_trials_half], ts[n_trials_half:2 * n_trials_half]):
            rows = idx[np.isin(trials[idx], h)]
            if len(rows) < n_rows:
                halves = None
                break
            halves.append(rng.choice(rows, n_rows, replace=False))
        if halves:
            out[int(s)] = tuple(halves)
    return out


def _between_subject_terms(X, halves):
    """Pairwise between-subject mean/cov terms, using each subject's FIRST half.

    Every group is (n_trials_half trials, n_rows rows) — structurally identical to the two
    halves the null floor compares, so the two are on the same scale and the subtraction is
    meaningful.
    """
    keys = sorted(halves)
    mt, ct = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            A, B = X[halves[keys[i]][0]], X[halves[keys[j]][0]]
            _, m, c = gaussian_kl_split(A, B, ridge=0.0)
            mt.append(m); ct.append(c)
    return np.asarray(mt), np.asarray(ct), len(keys)


def _null_terms(X, halves):
    """Estimation-noise floor: each subject's own two TRIAL-DISJOINT halves against each other.

    Two halves of one subject share a distribution by construction, so whatever the estimator
    reports here is sampling noise at the matched (trials, rows) budget. Anything below this
    floor is not distribution shift.
    """
    mt, ct = [], []
    for s in sorted(halves):
        h1, h2 = halves[s]
        _, m, c = gaussian_kl_split(X[h1], X[h2], ridge=0.0)
        mt.append(m); ct.append(c)
    if not mt:
        return float("nan"), float("nan"), 0
    return float(np.mean(mt)), float(np.mean(ct)), len(mt)


def _affine_invariance_check(frame, cap=400, seed=42):
    """Numerically assert the mean/cov split is invariant to a global affine map.

    This is the evidence that the original raw-vs-z-scored E3 contrast was not identifiable.
    Computed on the FULL basis with ridge=0 — the identity is algebraic and holds for any n>d.
    """
    rng = np.random.default_rng(seed)
    Xr = _repr_matrix(frame, "raw")
    Xg = _repr_matrix(frame, "pooled")
    subj = frame.subject.to_numpy()
    d = Xr.shape[1]
    cap = max(cap, d + 20)
    ok = [s for s in np.unique(subj) if (subj == s).sum() > d + 5]
    if len(ok) < 2:
        return dict(checked=False, reason=f"no two subjects with n > d={d}")
    s0, s1 = ok[0], ok[1]

    def take(X, s):
        idx = np.where(subj == s)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        return X[idx]

    # identical row selections for both representations (same rng draw per subject)
    i0 = take(np.arange(len(Xr)).reshape(-1, 1), s0).ravel()
    i1 = take(np.arange(len(Xr)).reshape(-1, 1), s1).ravel()

    def rel_diff(ridge):
        _, m_r, c_r = gaussian_kl_split(Xr[i0], Xr[i1], ridge=ridge)
        _, m_g, c_g = gaussian_kl_split(Xg[i0], Xg[i1], ridge=ridge)
        r = max(abs(m_r - m_g) / (abs(m_r) + 1e-12), abs(c_r - c_g) / (abs(c_r) + 1e-12))
        return (m_r, c_r, m_g, c_g, float(r))

    m_r, c_r, m_g, c_g, rel0 = rel_diff(0.0)          # the invariant quantity
    *_, rel_ridge = rel_diff(1e-3)                    # what the OLD code actually computed

    # The identity is exact ALGEBRA; the residual is pure floating-point. Its size is set by the
    # conditioning of the raw covariance, which `pinv` must invert. myobit (176 Hz -> 44-sample
    # windows -> near-degenerate features, cond ~1e13) lands at rel0 = 1.5e-3 with raw_mean_term
    # 31.35 vs 31.33 -- a 0.06 % disagreement, i.e. machine precision, not a broken identity.
    # So the tolerance is conditioning-aware: eps * cond is the achievable floor.
    S = np.cov(Xr[i0], rowvar=False)
    cond = float(np.linalg.cond(S))
    eps = float(np.finfo(np.float64).eps)
    tol = max(1e-6, min(1e-2, 100.0 * eps * cond))
    numerically_limited = bool(rel0 > 1e-3 and rel0 < tol)

    # Tolerance is 1e-3, not machine epsilon: raw sEMG feature scales span ~10^6, so the raw
    # covariance has condition number ~10^12 and `pinv` loses ~6 digits. The identity is exact
    # in exact arithmetic; what matters is that the residual is orders of magnitude smaller
    # than any effect one would report. `ridge_relative_difference` shows the contrast: the
    # +1e-3*I regulariser moves the split by O(0.1-1), i.e. 100-1000x the numerical floor —
    # every "raw vs z-scored" difference the original E3 reported came from there.
    return dict(checked=True, d=int(d), n_per_subject=int(len(i0)),
                raw_mean_term=float(m_r), raw_cov_term=float(c_r),
                global_z_mean_term=float(m_g), global_z_cov_term=float(c_g),
                max_relative_difference=float(rel0),
                ridge_relative_difference=float(rel_ridge),
                covariance_condition_number=cond,
                numerical_tolerance=float(tol),
                numerically_limited=numerically_limited,
                invariant=bool(rel0 < tol),
                ridge_breaks_invariance=bool(rel_ridge > 100 * max(rel0, 1e-12)),
                note=("The mean/cov split is invariant to a global affine map (verified at "
                      "ridge=0). A raw-vs-global-z-score contrast therefore measures nothing; "
                      "only a PER-SUBJECT map can change between-subject divergence. With the "
                      "old ridge=1e-3 the two disagree, which is what the original E3 reported."))


def meancov_decomposition(dataset, seed=42, cap=400, samples_per_dim=10):
    """E3 (rewritten). Between-subject KL mean/cov terms across three representations, each
    null-corrected against a within-subject split-half floor.

    The per-group sample size `n_group` is chosen so that (a) every subject can supply it,
    and (b) every subject can supply TWO disjoint groups of that size, which the split-half
    null floor requires. The null and the between-subject terms are then estimated at exactly
    the same n, which is what makes the subtraction valid. PCA is truncated to keep
    n_group / d >= `samples_per_dim`.
    """
    frame = windows.build_fast_frame(dataset, seed=seed)
    subj = frame.subject.to_numpy()
    trials = cv.trial_ids(frame)

    inv = _affine_invariance_check(frame, seed=seed)

    # Budget: every subject must supply two disjoint halves with the same trial and row count.
    per_subj_trials = np.array([len(np.unique(trials[subj == s])) for s in np.unique(subj)])
    n_trials_half = int(per_subj_trials.min() // 2)
    # rows available in the smaller half, worst case over subjects (approximate, then verified
    # exactly by `_matched_halves`, which drops any subject that cannot meet the budget)
    per_subj_rows = np.array([(subj == s).sum() for s in np.unique(subj)])
    n_rows = int(min(cap, (per_subj_rows.min() * n_trials_half) // max(per_subj_trials.min(), 1)))
    n_comp = max(2, n_rows // samples_per_dim)              # n/d >= samples_per_dim

    out = dict(affine_invariance_check=inv, n_pca_components=None,
               n_trials_per_half=n_trials_half, n_rows_per_group=n_rows, requested_cap=int(cap),
               min_trials_per_subject=int(per_subj_trials.min()),
               min_windows_per_subject=int(per_subj_rows.min()),
               design=("between-subject pairs and the within-subject null floor both compare "
                       "groups of exactly (n_trials_per_half trials, n_rows_per_group rows); "
                       "halves are TRIAL-disjoint so a trial's overlapping windows never span "
                       "both sides"))

    rng = np.random.default_rng(seed)
    halves = _matched_halves(subj, trials, n_trials_half, n_rows, rng) if (
        n_trials_half >= 2 and n_rows > n_comp + 5) else {}
    out["n_subjects_used"] = len(halves)
    out["n_subjects_dropped"] = int(len(per_subj_trials) - len(halves))
    if len(halves) < 2:
        out["representations"] = {r: dict(note=(f"need >=2 subjects with >= {2*n_trials_half} "
                                                f"trials and {2*n_rows} windows; got {len(halves)}"))
                                  for r in ("pooled", "subject_center", "subject_zscore")}
        out["warning"] = "insufficient trials/windows per subject for a matched null floor"
        return out

    reps = {}
    for rep in ("pooled", "subject_center", "subject_zscore"):
        X = _repr_matrix(frame, rep)
        Xp, d_eff = _pca_project(X, n_comp, seed)
        out["n_pca_components"] = d_eff
        mt, ct, n_sub = _between_subject_terms(Xp, halves)
        nm, nc, n_null = _null_terms(Xp, halves)
        n_drop = out["n_subjects_dropped"]
        n_eff = n_rows
        if mt.size == 0:
            reps[rep] = dict(note="fewer than 2 usable subjects")
            continue
        m, c = float(mt.mean()), float(ct.mean())
        have_null = (nm == nm and nc == nc and n_null > 0)
        me = max(0.0, m - nm) if have_null else float("nan")
        ce = max(0.0, c - nc) if have_null else float("nan")
        tot_e = (me + ce) if have_null else float("nan")
        # Detectability: if the excess over the estimation-noise floor is a small fraction of
        # that floor, there is no measurable between-subject shift in this representation, and
        # any "share" or "fraction removed" computed from it is a ratio of noise to noise.
        null_tot = (nm + nc) if have_null else float("nan")
        detectable = bool(have_null and tot_e > DETECT_FRAC * null_tot)
        reps[rep] = dict(
            n_subjects=int(n_sub), n_subjects_dropped_too_few_windows=n_drop,
            n_pairs=int(mt.size), n_per_group=int(n_rows),
            n_trials_per_group=int(n_trials_half),
            mean_term=m, cov_term=c, total_kl=m + c,
            null_mean_term=nm, null_cov_term=nc, null_total=null_tot,
            n_null_subjects=int(n_null), null_n_per_half=int(n_eff),
            null_is_trial_disjoint=True,
            null_estimated=bool(have_null),
            mean_term_excess=me, cov_term_excess=ce, total_kl_excess=tot_e,
            shift_detectable=detectable,
            snr_excess_over_null=float(tot_e / null_tot) if (have_null and null_tot > 0)
            else float("nan"),
            mean_share_of_excess=(float(me / tot_e) if (detectable and tot_e > 0)
                                  else float("nan")),
            # what you would have concluded WITHOUT subtracting the estimation-noise floor:
            # for two identical distributions this reads ~0.07, i.e. "covariance dominates".
            uncorrected_mean_share=float(m / (m + c)) if (m + c) > 0 else float("nan"),
        )
    out["representations"] = reps

    pooled = reps.get("pooled", {})
    p = pooled.get("total_kl_excess")
    for rep in ("subject_center", "subject_zscore"):
        q = reps.get(rep, {}).get("total_kl_excess")
        # Only meaningful when the POOLED representation had a detectable shift to remove.
        if pooled.get("shift_detectable") and p and p == p and q == q and p > 0:
            out[f"kl_excess_removed_by_{rep}"] = float(1.0 - q / p)
        else:
            out[f"kl_excess_removed_by_{rep}"] = None
    if not pooled.get("shift_detectable"):
        out["warning"] = ("no between-subject shift detectable above the estimation-noise floor "
                          "in the pooled representation; mean/cov shares are undefined here")
    out["interpretation"] = (
        "mean_share_of_excess in the `pooled` representation is the Yoneda quantity: the "
        "fraction of null-corrected between-subject Gaussian-KL attributable to mean shift. "
        "kl_excess_removed_by_subject_center is the divergence a per-subject mean re-estimation "
        "would remove — the cheap calibration that actually helps cross-subject.")
    return out


# --------------------------------------------------------------------------------------
# E4 — conditional-shift disparity
# --------------------------------------------------------------------------------------
def conditional_disparity(frame, seed=42, n_jobs=None):
    """E4. Conditional-shift disparity matrix: train kNN labeller on subject j, measure its
    label-disagreement on subject i; d_ij = min(err_ij, err_ji). Aggregate = rescaled Frobenius."""
    from sklearn.neighbors import KNeighborsClassifier
    from joblib import Parallel, delayed
    n_jobs = config.resolve_jobs() if n_jobs is None else n_jobs
    X = _basis(frame); y = frame.label.to_numpy(); subj = frame.subject.to_numpy()
    rng = np.random.default_rng(seed)
    cap = 400
    subs = [s for s in sorted(np.unique(subj)) if (subj == s).sum() >= 30]
    data = {}
    for s in subs:
        idx = np.where(subj == s)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        data[s] = (X[idx], y[idx])

    def err(a, b):   # kNN trained on b, error on a
        Xa, ya = data[a]; Xb, yb = data[b]
        if len(np.unique(yb)) < 2:
            return np.nan
        k = KNeighborsClassifier(n_neighbors=5).fit(Xb, yb)
        return float((k.predict(Xa) != ya).mean())

    pairs = [(i, j) for i in range(len(subs)) for j in range(i + 1, len(subs))]
    with progress.timer(f"E4 conditional disparity: {len(pairs)} pairs"):
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(lambda i, j: min(err(subs[i], subs[j]), err(subs[j], subs[i])))(i, j)
            for i, j in pairs)
    n = len(subs); D = np.zeros((n, n))
    for (i, j), v in zip(pairs, res):
        D[i, j] = D[j, i] = (v if v == v else 0.0)
    off = D[~np.eye(n, dtype=bool)]
    frob = (float(np.sqrt(((off / (off.max() + 1e-12)) ** 2).sum()) / len(off) ** 0.5)
            if off.size else float("nan"))
    # reference: a labeller that transfers nothing sits at 1 - 1/n_classes
    n_cls = int(len(np.unique(y)))
    return dict(n_subjects=n, conditional_disparity_frob=frob,
                mean_disagreement=float(off.mean()) if off.size else float("nan"),
                chance_disagreement=float(1.0 - 1.0 / n_cls), n_classes=n_cls)


def run(dataset, seed=42, n_jobs=None):
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "block_c"
    outdir.mkdir(parents=True, exist_ok=True)
    frame = windows.build_fast_frame(dataset, seed=seed)
    result = dict(
        dataset=dataset,
        E2_a4_fair=a4_fair(frame, dataset, seed),
        E3_meancov=meancov_decomposition(dataset, seed),
        E4_conditional_disparity=conditional_disparity(frame, seed, n_jobs),
    )
    (outdir / f"{dataset}__block_c.json").write_text(json.dumps(result, indent=2))
    return result
