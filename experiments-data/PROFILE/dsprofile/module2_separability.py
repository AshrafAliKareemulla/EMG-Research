"""Module 2 — class structure & separability + intrinsic dimensionality.

Filter indices (Fisher, Davies-Bouldin, silhouette, kNN-LOO), the distance-family
Mahalanobis separability index (DFSS, summary 08), the MI-family symmetrical-uncertainty
(CFSS, summary 08), and intrinsic dimensionality (PCA-95% + nonlinear TwoNN, summaries 06/14).
All computed on the representative de-duplicated basis, standardised per column.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, cv, windows


def _basis_matrix(frame):
    cols = []
    for fb in config.REPR_BASIS:
        cols += [c for c in frame.columns if c.startswith(fb + "_c")]
    X = frame[cols].to_numpy(dtype=np.float64)
    # standardise columns; drop degenerate/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, cols


def fisher_ratio(X, y):
    """Multi-class Fisher discriminant ratio = tr(S_b)/tr(S_w) averaged over features."""
    classes = np.unique(y)
    overall = X.mean(0)
    sb = sw = 0.0
    for c in classes:
        Xc = X[y == c]
        nc = len(Xc)
        mc = Xc.mean(0)
        sb += nc * np.sum((mc - overall) ** 2)
        sw += np.sum((Xc - mc) ** 2)
    return float(sb / (sw + 1e-12))


def mahalanobis_si(X, y):
    """DFSS-style separability: mean over class-pairs of Mahalanobis distance between centroids,
    using the POOLED WITHIN-CLASS covariance (not the total covariance, which would inflate the
    variance along the separation axis and shrink the distance). Higher = more separable (summary 08)."""
    classes = np.unique(y)
    d = X.shape[1]
    cov = np.zeros((d, d)); dof = 0
    for c in classes:
        Xc = X[y == c]
        if len(Xc) > 1:
            cov += (len(Xc) - 1) * np.cov(Xc, rowvar=False)
            dof += len(Xc) - 1
    cov = cov / max(dof, 1) + np.eye(d) * 1e-3
    inv = np.linalg.pinv(cov)
    cents = {c: X[y == c].mean(0) for c in classes}
    ds = []
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            d = cents[a] - cents[b]
            ds.append(np.sqrt(max(0.0, d @ inv @ d)))
    return float(np.mean(ds)) if ds else float("nan")


def _entropy(a):
    _, c = np.unique(np.asarray(a), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def _joint_entropy(a, b):
    ab = np.stack([np.asarray(a), np.asarray(b)], axis=1)
    _, c = np.unique(ab, axis=0, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def _quantile_bin(x, bins=10):
    """Equal-frequency discretisation of a continuous feature.

    A low-cardinality feature is passed through UNBINNED. Quantile-binning it would MERGE distinct
    values into one bin and destroy information: a 3-valued feature pushed through 10 quantile
    edges collapses to 2 bins, so even an exact copy of a 3-class label scored SU=0.72 instead of
    1.0 (caught by tests/test_fixes_20260712.py).
    """
    x = np.asarray(x, float)
    u = np.unique(x)
    if len(u) <= bins:
        return np.searchsorted(u, x)                # use the values themselves as categories
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.zeros(len(x), dtype=int)          # constant/degenerate feature
    return np.digitize(x, edges[1:-1])


def mi_symmetric_uncertainty(X, y, cols, bins=10):
    """CFSS-style symmetrical uncertainty SU = 2 I(C;F) / (H(C) + H(F)) per feature (summary 08).

    F1 (fixed 2026-07-12). Two defects, both now closed:

    1. The old code divided by **H(C) alone**. H(C) is constant across features, so the ranking
       collapsed to a raw-MI ranking (H(F), which differs per feature, was dropped), and the value
       could exceed 1 — impossible for an uncertainty coefficient (an exact copy of the label
       scored ~2.0).
    2. The obvious repair — keep sklearn's kNN `mutual_info_classif` for I(C;F) but bin the feature
       for H(F) — is ALSO wrong: the two estimators are inconsistent, so an exact label copy scored
       1.28 (verified in tests/test_fixes_20260712.py before this rewrite). I(C;F), H(C) and H(F)
       must all come from the SAME discretisation.

    So: quantile-bin the feature, then use the discrete plug-in identity
    I(C;F) = H(C) + H(F) - H(C,F). Ground truth: feature == label -> SU == 1.0 exactly;
    feature independent of label -> SU == 0; every SU lies in [0, 1].
    """
    X = np.asarray(X, float)
    hc = _entropy(y)
    out = {}
    for j, c in enumerate(cols):
        fb = _quantile_bin(X[:, j], bins)
        hf = _entropy(fb)
        mi = hc + hf - _joint_entropy(y, fb)
        out[c] = float(np.clip(2.0 * mi / (hc + hf + 1e-12), 0.0, 1.0))
    return out


# NOTE: the former `knn_loo` was neither leave-one-out nor leakage-safe — it shuffled rows
# and ran a plain 5-fold, so 50 %-overlapping windows from one trial straddled the fold
# boundary. Use `cv.knn_trial_cv` (within-subject, trial-grouped) and `cv.knn_loso`
# (subject-disjoint) instead; both are reported so the gap between them is visible.


def twonn_dim(X, max_n=3000, seed=0):
    """Two-NN intrinsic-dimension estimator (Facco et al.)."""
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    mask = r1 > 1e-12
    mu = (r2[mask] / r1[mask])
    mu = mu[mu > 1.0]
    if len(mu) < 10:
        return float("nan")
    x = np.sort(np.log(mu))                          # log(mu) sorted ascending
    n = len(x)
    F = np.arange(1, n + 1) / n                       # empirical CDF at each rank
    # drop the last point (F=1 -> -log(0)=inf); fit y = d*x through the origin (Facco et al.)
    x = x[:-1]; y = -np.log(1.0 - F[:-1])
    d_est = np.sum(x * y) / (np.sum(x * x) + 1e-12)
    return float(d_est)


def pca95(X):
    from sklearn.decomposition import PCA
    n = min(X.shape[0] - 1, X.shape[1])
    p = PCA(n_components=n).fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    return int(np.searchsorted(cum, 0.95) + 1)


def run(dataset, seed=42):
    config.ensure_dirs()
    frame = windows.build_fast_frame(dataset, seed=seed)
    X, cols = _basis_matrix(frame)
    y = frame.label.to_numpy()

    from sklearn.metrics import davies_bouldin_score, silhouette_score
    rng = np.random.default_rng(seed)
    sidx = rng.choice(len(X), min(4000, len(X)), replace=False)   # silhouette is O(n^2)

    groups = cv.trial_ids(frame)
    subjects = frame.subject.to_numpy()
    knn_trial = cv.knn_trial_cv(X, y, groups, seed=seed)
    knn_xsub = cv.knn_loso(X, y, subjects, seed=seed)

    result = dict(
        dataset=dataset,
        n_windows=int(len(X)), n_classes=int(len(np.unique(y))), n_features=len(cols),
        fisher_ratio=fisher_ratio(X, y),
        davies_bouldin=float(davies_bouldin_score(X, y)),
        silhouette=float(silhouette_score(X[sidx], y[sidx])),
        # within-subject, trial-grouped (no overlapping-window leak)
        knn_trial_cv_acc=knn_trial,
        # subject-disjoint: the cross-subject number the paper's thesis is about
        knn_loso_acc=knn_xsub,
        # how much of the "separability" was within-subject structure
        within_minus_cross=float(knn_trial - knn_xsub)
        if (knn_trial == knn_trial and knn_xsub == knn_xsub) else float("nan"),
        mahalanobis_si=mahalanobis_si(X, y),
        pca95_dim=pca95(X),
        twonn_dim=twonn_dim(X, seed=seed),
        n_channels=int(frame.attrs["n_channels"]),
        protocol_note=("knn_trial_cv_acc: GroupKFold on trial (within-subject). "
                       "knn_loso_acc: GroupKFold on subject. The pre-2026-07-10 `knn_loo_acc` "
                       "was a shuffled 5-fold over overlapping windows and is superseded."),
    )
    su = mi_symmetric_uncertainty(X, y, cols)
    result["mi_su_top"] = dict(sorted(su.items(), key=lambda kv: -kv[1])[:10])

    outdir = config.RESULTS_DIR / "module2"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{dataset}__separability.json").write_text(json.dumps(result, indent=2))
    return result
