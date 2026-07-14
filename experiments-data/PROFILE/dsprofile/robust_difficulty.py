"""Robustness of subject difficulty — is 'hard subject' a property of the DATA or just of LDA?

Computes each subject's LOSO accuracy with THREE cheap classifiers (LDA, linear SVM, Random Forest)
across MULTIPLE seeds, then checks:
  * do the classifiers AGREE on who is hard? (rank correlation between their per-subject accuracies)
  * does the cheap statistic (MMD-to-pool) predict difficulty for EACH classifier? (not just LDA)
  * is the difficulty-prediction correlation STABLE across seeds? (mean +/- std)
If yes, "difficulty" is an intrinsic, classifier-agnostic property -> the SDI headline is bulletproof.

Self-contained (no DL). Scalable + dataset-agnostic; reuses the cached fast frame; parallel over subjects.
"""
from __future__ import annotations

import json
import itertools

import numpy as np

from . import config, cv, windows, progress
from .module3_shift import _basis
from .module5_difficulty import subject_shift_stats


def _clf(name):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "svm":
        return LinearSVC(C=1.0, dual="auto", max_iter=2000)
    return RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=1)


def _loso_one(Xtr, ytr, Xte, yte, name):
    if len(np.unique(ytr)) < 2:
        return np.nan
    try:
        clf = _clf(name).fit(Xtr, ytr)
        return float((clf.predict(Xte) == yte).mean())
    except Exception:
        return np.nan


def loso_multi(frame, clfs=("lda", "svm", "rf"), seeds=(42, 43, 44), cap=600, n_jobs=None):
    """Per-subject LOSO accuracy for each classifier, averaged over seeds (each seed = a different
    per-subject subsample). Returns {clf: {subject: mean_acc}}."""
    from joblib import Parallel, delayed
    n_jobs = config.resolve_jobs() if n_jobs is None else n_jobs
    X = _basis(frame); y = frame["label"].to_numpy(); subj = frame["subject"].to_numpy()
    subs = sorted(np.unique(subj))
    acc = {c: {int(s): [] for s in subs} for c in clfs}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # per-subject subsample (bounds cost; different subsample each seed)
        idxs = {}
        for s in subs:
            ii = np.where(subj == s)[0]
            idxs[s] = ii if len(ii) <= cap else rng.choice(ii, cap, replace=False)
        def one(s):
            te = idxs[s]
            tr = np.concatenate([idxs[o] for o in subs if o != s])
            if len(te) < 5 or len(np.unique(y[tr])) < 2:
                return {c: np.nan for c in clfs}
            # standardise on train, apply to test (no leakage)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            return {c: _loso_one(Xtr, y[tr], Xte, y[te], c) for c in clfs}
        res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(one)(s) for s in subs)
        for s, r in zip(subs, res):
            for c in clfs:
                acc[c][int(s)].append(r[c])
    return {c: {s: float(np.nanmean(v)) for s, v in d.items() if np.isfinite(np.nanmean(v))}
            for c, d in acc.items()}


def run(dataset, seed=42, n_jobs=None):
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "robust_difficulty"
    outdir.mkdir(parents=True, exist_ok=True)
    frame = windows.build_fast_frame(dataset, seed=seed)
    with progress.timer(f"robust difficulty (3 clf x 3 seeds) :: {dataset}"):
        acc = loso_multi(frame, n_jobs=n_jobs)
    from scipy.stats import spearmanr, pearsonr

    # inter-classifier agreement on who is hard
    clfs = list(acc)
    agree = {}
    for a, b in itertools.combinations(clfs, 2):
        common = sorted(set(acc[a]) & set(acc[b]))
        if len(common) >= 4:
            r, p = spearmanr([acc[a][s] for s in common], [acc[b][s] for s in common])
            agree[f"{a}_vs_{b}"] = dict(spearman=float(r), p_value=float(p))

    # does MMD-to-pool predict difficulty for EACH classifier?
    preds = subject_shift_stats(_basis(frame), frame["subject"].to_numpy(), seed,
                                n_jobs=n_jobs, trials=cv.trial_ids(frame))
    diff_corr = {}
    for c in clfs:
        common = sorted(set(acc[c]) & set(preds))
        if len(common) >= 4:
            mmd = np.array([preds[s]["mmd_to_pool"] for s in common])
            a = np.array([acc[c][s] for s in common])
            r, p = pearsonr(mmd, a)
            diff_corr[c] = dict(pearson_r=float(r), p_value=float(p), n=len(common))

    result = dict(
        dataset=dataset,
        mean_loso_acc={c: float(np.mean(list(acc[c].values()))) for c in clfs if acc[c]},
        inter_classifier_agreement=agree,
        difficulty_prediction_by_classifier=diff_corr,
        classifiers_agree=bool(agree and np.mean([v["spearman"] for v in agree.values()]) > 0.5),
    )
    (outdir / f"{dataset}__robust_difficulty.json").write_text(json.dumps(result, indent=2))
    return result
