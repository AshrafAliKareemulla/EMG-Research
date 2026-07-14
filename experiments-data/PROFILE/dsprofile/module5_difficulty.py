"""Module 5 — subject-difficulty predictor (the money result).

SELF-CONTAINED. Everything here is computed from this repo's own features. Paper 2 has NO
dependency on any other track and involves no deep learning.

The TARGET is each subject's leave-one-subject-out (LOSO) accuracy using an LDA on the
representative basis. The PREDICTORS are cheap per-subject distribution-shift statistics to the
training pool (MMD, H-divergence, Gaussian-KL mean/cov terms). We report how well the cheap
statistics predict LOSO accuracy (Pearson r + an out-of-sample R^2) — the Albuquerque et al.
(summary 07) result, extended to sEMG across many datasets.

"Is this just an LDA predicting an LDA?" is answered INSIDE Paper 2: `robust_difficulty` shows
LDA/SVM/RF agree on who is hard, and X2/X6 show the correlation survives a disjoint feature basis
and a learned embedding.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, cv, windows, progress
from .module3_shift import mmd_rbf, h_divergence, gaussian_kl_split, _basis


def loso_lda_accuracy(X, y, subjects):
    """Per-subject LOSO accuracy with LDA (subject-disjoint train/test).

    Standardisation is refit on the training subjects only. (LDA is affine-invariant so this
    changes nothing numerically, but the roadmap promised a train-only fit and the same helper
    is reused by scale-sensitive classifiers.)
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    accs = {}
    subs = sorted(np.unique(subjects))
    for s in subs:
        tr = subjects != s; te = subjects == s
        if len(np.unique(y[tr])) < 2 or te.sum() < 5:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LinearDiscriminantAnalysis()
        clf.fit((X[tr] - mu) / sd, y[tr])
        accs[int(s)] = float((clf.predict((X[te] - mu) / sd) == y[te]).mean())
    return accs


def _subj_stat(this, rest, seed, t_this=None, t_rest=None):
    """One subject's distance-to-pool statistics (runs in a joblib worker)."""
    rng = np.random.default_rng(seed)
    _, kl_mean, kl_cov = gaussian_kl_split(this, rest)
    return dict(mmd_to_pool=mmd_rbf(this, rest, rng=rng),
                hdiv_to_pool=h_divergence(this, rest, rng=rng,
                                          groups_a=t_this, groups_b=t_rest),
                kl_mean_to_pool=kl_mean, kl_cov_to_pool=kl_cov)


def subject_shift_stats(X, subjects, seed, n_jobs=-1, cap=800, trials=None):
    """Per-subject distance-to-pool predictors, parallelised across subjects.

    `trials` (per-row trial ids) is REQUIRED for a trustworthy `hdiv_to_pool`: without it the
    H-divergence classifier is cross-validated over shuffled, 50 %-overlapping windows and
    memorises trial identity, saturating d_H near its maximum of 2 regardless of the true
    divergence. Pass `cv.trial_ids(frame)`.

    Each subject's own windows and the pool are capped to `cap` rows (bounds worker memory;
    the distance functions subsample internally anyway).
    """
    from joblib import Parallel, delayed
    rng = np.random.default_rng(seed)
    if trials is None:
        progress.log("WARNING subject_shift_stats without trial ids -> hdiv_to_pool is leaked")
    trials = None if trials is None else np.asarray(trials)
    subs = [int(s) for s in sorted(np.unique(subjects))]
    jobs, valid = [], []
    for k, s in enumerate(subs):
        i_this = np.where(subjects == s)[0]
        i_rest = np.where(subjects != s)[0]
        if len(i_this) < 20 or len(i_rest) < 20:
            continue
        if len(i_this) > cap:
            i_this = rng.choice(i_this, cap, replace=False)
        if len(i_rest) > cap:
            i_rest = rng.choice(i_rest, cap, replace=False)
        t_this = None if trials is None else trials[i_this]
        t_rest = None if trials is None else trials[i_rest]
        jobs.append(delayed(_subj_stat)(X[i_this], X[i_rest], seed + k, t_this, t_rest))
        valid.append(s)
    with progress.timer(f"module5 per-subject stats ({len(valid)} subjects)"):
        out = Parallel(n_jobs=n_jobs, backend="loky")(jobs)
    return {s: r for s, r in zip(valid, out)}


def run(dataset, seed=42, n_jobs=None):
    config.ensure_dirs()
    n_jobs = config.resolve_jobs() if n_jobs is None else n_jobs
    outdir = config.RESULTS_DIR / "module5"
    outdir.mkdir(parents=True, exist_ok=True)
    frame = windows.build_fast_frame(dataset, seed=seed)
    X = _basis(frame)
    y = frame.label.to_numpy()
    subjects = frame.subject.to_numpy()

    # --- difficulty TARGET -------------------------------------------------------------
    # SELF-CONTAINED: the target is computed HERE, from this repo's own features. No external
    # dependency, no deep learning. "Is this just an LDA predicting an LDA?" is answered inside
    # Paper 2 by `robust_difficulty` (LDA/SVM/RF agree on who is hard) and by X2/X6 (the
    # correlation survives a disjoint feature basis and a learned embedding).
    with progress.timer(f"module5 {dataset} self-LDA-LOSO target"):
        target = loso_lda_accuracy(X, y, subjects)
    target_source = "self_lda_loso"

    preds = subject_shift_stats(X, subjects, seed, n_jobs=n_jobs, trials=cv.trial_ids(frame))
    common = sorted(set(target) & set(preds))
    if len(common) < 4:
        result = dict(dataset=dataset, note="too few subjects with both target and predictors",
                      n_common=len(common), target_source=target_source)
    else:
        acc = np.array([target[s] for s in common])
        pred_names = ["mmd_to_pool", "hdiv_to_pool", "kl_mean_to_pool", "kl_cov_to_pool"]
        P = {p: np.array([preds[s][p] for s in common]) for p in pred_names}
        corr = {}
        from scipy.stats import pearsonr
        for p in pred_names:
            r, pv = pearsonr(P[p], acc)
            corr[p] = dict(pearson_r=float(r), p_value=float(pv))

        # --- combined model: LOSO-CV R^2, not in-sample -------------------------------
        # The old `combined_linear_r2` was LinearRegression().score() on the SAME rows it was
        # fit on: 4 predictors, n=25 subjects -> R^2=0.61 for emaha_db1 while the best single
        # predictor only reached r=-0.43 (r^2=0.18). That is fit, not prediction. We now
        # leave one SUBJECT out, refit, and score out-of-sample. A negative value (worse than
        # predicting the mean) is a legitimate outcome and is reported as such.
        from sklearn.linear_model import LinearRegression
        M = np.column_stack([P[p] for p in pred_names])
        r2_in = float(LinearRegression().fit(M, acc).score(M, acc))
        oof = np.empty_like(acc)
        for i in range(len(acc)):
            tr = np.ones(len(acc), bool); tr[i] = False
            oof[i] = LinearRegression().fit(M[tr], acc[tr]).predict(M[i:i + 1])[0]
        ss_res = float(((acc - oof) ** 2).sum())
        ss_tot = float(((acc - acc.mean()) ** 2).sum())
        r2_cv = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        # LOO-CV of a 4-predictor OLS on n-1 training subjects is wildly unstable when n is
        # small: the 2026-07-10 run produced R^2 = -278 (emaha_db5, n=10) and -198 (emaha_db7).
        # Those are not estimates. Report the value but mark it unusable below ~5 subjects per
        # predictor; the single-predictor `primary_pearson_r` remains valid at any n.
        min_n_for_cv_r2 = 5 * len(pred_names)
        cv_r2_reliable = bool(len(common) >= min_n_for_cv_r2)

        prim = config.PRIMARY_PREDICTOR
        result = dict(
            dataset=dataset, target_source=target_source, n_subjects=len(common),
            loso_acc_mean=float(acc.mean()), loso_acc_std=float(acc.std()),
            predictor_correlations=corr,
            # fixed a priori: quoting the best-of-4 per dataset is a winner's curse over
            # 4 predictors x 14 datasets = 56 uncorrected tests. FDR is applied in meta.py.
            primary_predictor=prim,
            primary_pearson_r=corr[prim]["pearson_r"],
            primary_p_value=corr[prim]["p_value"],
            combined_cv_r2=(r2_cv if cv_r2_reliable else None),
            combined_cv_r2_raw=r2_cv,
            combined_cv_r2_reliable=cv_r2_reliable,
            combined_cv_r2_min_subjects=min_n_for_cv_r2,
            combined_insample_r2=r2_in,
            r2_note=("combined_cv_r2 is leave-one-subject-out out-of-sample R^2 and is the only "
                     "one that may be quoted as predictive power; combined_insample_r2 is kept "
                     "for comparison and is optimistic by construction."),
            best_predictor_posthoc=min(corr, key=lambda k: corr[k]["p_value"]),
            best_predictor_warning="post-hoc selected; do not quote without FDR correction",
        )
        pd.DataFrame({"subject": common, "loso_acc": acc,
                      **{p: P[p] for p in pred_names}}).to_parquet(
            outdir / f"{dataset}__difficulty.parquet")

    (outdir / f"{dataset}__difficulty.json").write_text(json.dumps(result, indent=2))
    return result
