"""Leakage-safe cross-validation helpers.

WHY THIS EXISTS (2026-07-10 results audit)
------------------------------------------
Every kNN "accuracy" in Phase 1/2 was computed as::

    idx = rng.choice(len(X), max_n, replace=False)      # shuffles row order
    cross_val_score(KNeighborsClassifier(5), X[idx], y[idx], cv=5)

Windows overlap 50 %, so a test window's nearest neighbour is very often its own
overlapping sibling from the same trial, sitting in the training fold. The split was also
subject-pooled. The resulting numbers were inflated (emaha_db1: kNN 0.42 vs a real LOSO of
0.25) and they fed Module 2's `knn_loo_acc`, Block B's class ranking, Block D's channel /
sampling-rate curves and — via `knn_loo` — the meta-analysis headline.

Two honest protocols are provided instead, and every caller reports BOTH:

* ``trial_cv``   — GroupKFold grouped on trial. Within-subject, but no window from a trial
                   is ever split across train/test. This is a *within-subject* separability
                   measure and must be labelled as such.
* ``loso``       — GroupKFold grouped on subject (i.e. leave-one-subject-out when
                   n_splits == n_subjects). This is the cross-subject number the paper's
                   thesis is actually about.

Standardisation is fit on the TRAINING fold only, inside the loop.
"""
from __future__ import annotations

import numpy as np


META_KEYS = ("subject", "session", "label", "repetition")


def trial_ids(frame) -> np.ndarray:
    """A unique integer id per recording trial = (subject, session, label, repetition).

    Windows carved from one trial share an id, so grouping on it prevents a window and its
    50 %-overlapping neighbour landing on opposite sides of a fold boundary.
    """
    missing = [k for k in META_KEYS if k not in frame.columns]
    if missing:
        raise KeyError(f"frame lacks meta columns {missing}; cannot build trial ids")
    keys = np.column_stack([frame[k].to_numpy() for k in META_KEYS])
    _, ids = np.unique(keys, axis=0, return_inverse=True)
    return ids.astype(np.int64)


def _subsample_by_group(groups, y, max_n, seed):
    """Subsample rows to <= max_n while keeping EVERY group represented.

    An earlier version selected whole groups until the row budget filled. That was doubly
    wrong. It was unnecessary — `GroupKFold` assigns folds by group id, so a row-level
    subsample can never place a trial on both sides of a fold boundary — and it was
    catastrophic when the grouping variable is `subject`: with ~3.6k-9k windows per subject
    and max_n=4000 it retained ONE OR TWO subjects, so `knn_loso` silently trained on a
    single subject (or returned NaN at n_splits=1).

    Rows are drawn proportionally within each group, with at least one row per group, so both
    the group set and (approximately) the class balance are preserved.
    """
    n = len(groups)
    if max_n is None or n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    n_groups = len(uniq)
    if max_n < n_groups:                       # cannot keep every group; fall back to 1 each
        keep_g = rng.choice(n_groups, max_n, replace=False)
        return np.array([rng.choice(np.where(inv == g)[0]) for g in keep_g])
    # proportional allocation, floor 1, capped by the group's own size
    quota = np.maximum(1, np.floor(counts * max_n / n).astype(int))
    quota = np.minimum(quota, counts)
    keep = []
    for g in range(n_groups):
        idx = np.where(inv == g)[0]
        keep.append(idx if quota[g] >= len(idx) else rng.choice(idx, quota[g], replace=False))
    return np.sort(np.concatenate(keep))


def _grouped_cv_score(X, y, groups, n_splits, clf_factory, standardise=True):
    from sklearn.model_selection import GroupKFold
    n_groups = len(np.unique(groups))
    n_splits = int(min(n_splits, n_groups))
    if n_splits < 2:
        return float("nan")
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        Xtr, Xte = X[tr], X[te]
        if standardise:                                   # fit on TRAIN fold only
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        clf = clf_factory().fit(Xtr, y[tr])
        accs.append(float((clf.predict(Xte) == y[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def _knn(k):
    from sklearn.neighbors import KNeighborsClassifier
    return lambda: KNeighborsClassifier(n_neighbors=k)


def knn_trial_cv(X, y, groups, k=5, n_splits=5, max_n=4000, seed=0):
    """Within-subject separability: 5-fold kNN, folds grouped on trial (no window leak)."""
    sel = _subsample_by_group(groups, y, max_n, seed)
    return _grouped_cv_score(X[sel], y[sel], groups[sel], n_splits, _knn(k))


def knn_subject_cv(X, y, subjects, k=5, n_splits=5, max_n=4000, seed=0):
    """Cross-subject separability: kNN with SUBJECT-DISJOINT folds.

    This is subject-grouped `n_splits`-fold (leave-~N/5-subjects-out), **not** leave-one-
    subject-out: it is named accordingly, because a `knn_loso` that is really 5-fold would
    overpromise. It is the cheap cross-subject counterpart to `knn_trial_cv`; the true
    per-subject LOSO numbers live in `module5.loso_lda_accuracy` and `robust_difficulty`.

    Every subject is retained by the subsampler (see `_subsample_by_group`); the row budget is
    spread across them proportionally.
    """
    sel = _subsample_by_group(subjects, y, max_n, seed)
    return _grouped_cv_score(X[sel], y[sel], subjects[sel], n_splits, _knn(k))


# Back-compat alias. Prefer the honest name.
knn_loso = knn_subject_cv


def knn_predict_trial_cv(X, y, groups, k=5, n_splits=5, max_n=6000, seed=0):
    """Out-of-fold predictions under trial-grouped CV. Returns (y_true, y_pred)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    sel = _subsample_by_group(groups, y, max_n, seed)
    X, y, groups = X[sel], y[sel], groups[sel]
    n_splits = int(min(n_splits, len(np.unique(groups))))
    if n_splits < 2:
        return y, np.full_like(y, -1)
    pred = np.full_like(y, -1)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = KNeighborsClassifier(n_neighbors=k).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = clf.predict((X[te] - mu) / sd)
    ok = pred != -1
    return y[ok], pred[ok]
