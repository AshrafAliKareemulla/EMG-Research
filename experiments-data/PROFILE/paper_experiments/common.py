"""common.py — the hardened foundation for the Paper-2 experiment suite.

Pure (h5-free) building blocks: logging/timing, atomic IO, numerically-safe math, statistics
(correlation with guards, FDR, random-effects pooling, partial correlation, cluster bootstrap,
permutation test), distribution distances (MMD with median-heuristic / multi-kernel, energy,
Gaussian-KL split, leak-safe H-divergence), representation transforms (per-subject centre/zscore,
CORAL, PCA & random-Fourier embeddings), leakage-safe LOSO classifiers, frame utilities, a dataset
runner (atomic / resume / shard), and a parametric synthetic-frame generator for ground truth.

Nothing here imports h5py or `semg`; `dsprofile.windows` is imported lazily inside `build_frame`.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

# --- make `dsprofile` importable regardless of cwd (paper_experiments/ lives under PROFILE) ------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROFILE = os.path.dirname(_HERE)
if _PROFILE not in sys.path:
    sys.path.insert(0, _PROFILE)

from dsprofile import config  # noqa: E402  (safe: config only sets paths/constants, no h5)

EPS = 1e-12
REPR_BASIS = list(config.REPR_BASIS)
AMPLITUDE_FEATURES = ["MAV", "WL", "RMS", "IEMG", "SSI", "DASDV", "AAC", "LOG", "LOGRMS", "NLE", "MFL"]
SHAPE_FEATURES = ["HJ_MOB", "HJ_COM", "WAMP", "ZC", "SSC", "MYOP", "SKEW", "KURT",
                  "MNF", "MDF", "SENT"]   # amplitude-invariant / shape-only families


# =====================================================================================
# 1. Logging & timing
# =====================================================================================
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    log(f"START {label}")
    try:
        yield
    finally:
        log(f"DONE  {label}  ({time.perf_counter() - t0:.1f}s)")


# =====================================================================================
# 2. Atomic IO & result dirs
# =====================================================================================
def results_dir(name: str):
    d = config.RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def atomic_write_json(path, obj) -> None:
    """Write JSON via a temp file + os.replace so a reader/collector never sees a partial file."""
    path = str(path)
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# =====================================================================================
# 3. Numerically-safe math
# =====================================================================================
def safe_div(a, b, default=0.0):
    """Elementwise a/b with a guarded denominator; NaN/inf -> default. Handles huge & tiny values."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = a / np.where(np.abs(b) < EPS, np.nan, b)
    return np.where(np.isfinite(out), out, default)


def zscore(X, axis=0, ddof=0):
    """Column z-score with a guarded std (constant columns -> 0, never inf)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=axis, keepdims=True)
    sd = X.std(axis=axis, ddof=ddof, keepdims=True)
    return (X - mu) / np.where(sd < EPS, 1.0, sd)


def finite_mask(*arrays):
    """Boolean mask where ALL given 1-D arrays are finite (drops NaN/inf jointly)."""
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def clip_corr(r, lim=0.999999):
    """Keep a correlation strictly inside (-1, 1) so arctanh / t-stats never blow up."""
    if r != r:
        return r
    return float(max(-lim, min(lim, r)))


# =====================================================================================
# 4. Statistics (all guarded; unit-tested against known values)
# =====================================================================================
def pearson(x, y):
    """Pearson r + two-sided p + n, on the jointly-finite subset. Returns (nan,nan,n) if degenerate."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = finite_mask(x, y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or x.std() < EPS or y.std() < EPS:
        return float("nan"), float("nan"), int(n)
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    return float(r), float(p), int(n)


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = finite_mask(x, y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or x.std() < EPS or y.std() < EPS:
        return float("nan"), float("nan"), int(n)
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p), int(n)


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg step-up FDR. Returns (rejected_bool, q_values) aligned to input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    q = np.full(n, np.nan)
    rej = np.zeros(n, bool)
    ok = np.where(np.isfinite(p))[0]
    if ok.size == 0:
        return rej, q
    order = ok[np.argsort(p[ok])]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    rej[order] = q[order] <= alpha
    return rej, q


def pool_random_effects(rs, ns):
    """DerSimonian-Laird random-effects pooling of correlations via the Fisher-z transform.

    Guards: clips r off +-1, drops n<4 (variance 1/(n-3) undefined), handles k<2.
    """
    rs = np.asarray(rs, float)
    ns = np.asarray(ns, float)
    keep = np.isfinite(rs) & np.isfinite(ns) & (ns >= 4)
    rs, ns = rs[keep], ns[keep]
    if len(rs) < 2:
        return dict(k=int(len(rs)), note="need >=2 datasets with n>=4 to pool")
    z = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    w = ns - 3.0
    sw = w.sum()
    z_fixed = float((w * z).sum() / sw)
    Q = float((w * (z - z_fixed) ** 2).sum())
    df = len(z) - 1
    C = sw - (w ** 2).sum() / sw
    tau2 = max(0.0, (Q - df) / C) if C > EPS else 0.0
    w_re = 1.0 / (1.0 / w + tau2)
    z_re = float((w_re * z).sum() / w_re.sum())
    se = math.sqrt(1.0 / w_re.sum())
    return dict(k=int(len(rs)), pooled_r=float(np.tanh(z_re)),
                ci95=[float(np.tanh(z_re - 1.96 * se)), float(np.tanh(z_re + 1.96 * se))],
                tau2=float(tau2), Q=Q, df=int(df),
                I2=float(max(0.0, (Q - df) / Q) if Q > EPS else 0.0),
                pooled_r_fixed=float(np.tanh(z_fixed)))


def partial_corr(x, y, z):
    """Partial correlation of x,y controlling for z (Frisch-Waugh-Lovell residualisation)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    m = finite_mask(x, y, z)
    x, y, z = x[m], y[m], z[m]
    if len(x) < 4:
        return float("nan")
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if rx.std() < EPS or ry.std() < EPS:
        return float("nan")
    return clip_corr(float(np.corrcoef(rx, ry)[0, 1]))


def cluster_bootstrap(rows, stat_fn, cluster_key, B=2000, seed=0):
    """Cluster (block) bootstrap: resample CLUSTERS with replacement, recompute `stat_fn(rows)`.

    `rows` is a list of dicts; `cluster_key` names the clustering column (e.g. 'dataset'). This is
    the honest inference when observations within a cluster are correlated (the fix for the
    nested-rung pseudo-replication). Returns dict(point, ci95, n_boot, excludes_zero).
    """
    rng = np.random.default_rng(seed)
    clusters = sorted({r[cluster_key] for r in rows})
    by = {c: [r for r in rows if r[cluster_key] == c] for c in clusters}
    point = stat_fn(rows)
    vals = []
    for _ in range(B):
        samp = rng.choice(clusters, len(clusters), replace=True)
        res = [r for c in samp for r in by[c]]
        v = stat_fn(res)
        if v == v:
            vals.append(v)
    if not vals:
        return dict(point=point, ci95=[float("nan")] * 2, n_boot=0, excludes_zero=False)
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return dict(point=float(point) if point == point else float("nan"),
                ci95=[lo, hi], n_boot=len(vals), n_clusters=len(clusters),
                excludes_zero=bool(lo > 0 or hi < 0))


def permutation_corr(x, y, B=20000, seed=0):
    """Two-sided permutation p for |pearson(x,y)|: P(|r_perm| >= |r_obs|), add-one smoothed."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = finite_mask(x, y)
    x, y = x[m], y[m]
    if len(x) < 4 or x.std() < EPS or y.std() < EPS:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt((xc ** 2).sum() * (yc ** 2).sum()) + EPS
    obs = float((xc * yc).sum() / denom)
    aobs = abs(obs)
    cnt = 0
    n = len(y)
    for _ in range(B):
        rp = abs(float((xc * yc[rng.permutation(n)]).sum() / denom))
        if rp >= aobs - 1e-12:
            cnt += 1
    return obs, float((cnt + 1) / (B + 1))


# =====================================================================================
# 5. Distribution distances
# =====================================================================================
def _subsample(X, n, rng):
    X = np.asarray(X, float)
    return X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]


def median_gamma(A, B, rng, n=200):
    """Median-heuristic RBF bandwidth: gamma = 1/(2 * median pairwise squared distance)."""
    from scipy.spatial.distance import pdist
    Z = np.vstack([_subsample(A, n, rng), _subsample(B, n, rng)])
    d2 = pdist(Z, "sqeuclidean")
    med = float(np.median(d2)) if d2.size else 0.0
    return 1.0 / (2.0 * med) if med > EPS else 1.0 / max(1, Z.shape[1])


def mmd_rbf(A, B, gamma="median", n=400, rng=None, kernels=None):
    """Biased RBF-MMD^2 (>=0). `gamma`: 'median' (heuristic), '1/d' (legacy sklearn default), or a
    float. `kernels`: an iterable of gammas -> multi-kernel MMD (mean over kernels)."""
    rng = rng or np.random.default_rng(0)
    A = _subsample(A, n, rng)
    B = _subsample(B, n, rng)
    if A.shape[0] < 2 or B.shape[0] < 2:
        return float("nan")
    from sklearn.metrics.pairwise import rbf_kernel
    if kernels is not None:
        gs = [g for g in kernels]
    elif gamma == "median":
        gs = [median_gamma(A, B, rng)]
    elif gamma == "1/d":
        gs = [1.0 / A.shape[1]]
    else:
        gs = [float(gamma)]
    vals = []
    for g in gs:
        Kaa = rbf_kernel(A, A, g).mean()
        Kbb = rbf_kernel(B, B, g).mean()
        Kab = rbf_kernel(A, B, g).mean()
        vals.append(max(0.0, Kaa + Kbb - 2 * Kab))
    return float(np.mean(vals))


def energy_distance(A, B, n=400, rng=None):
    rng = rng or np.random.default_rng(0)
    A = _subsample(A, n, rng)
    B = _subsample(B, n, rng)
    if len(A) < 2 or len(B) < 2:
        return float("nan")
    from scipy.spatial.distance import cdist
    return float(max(0.0, 2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean()))


def gaussian_kl_split(A, B, ridge=0.0):
    """KL(N_A||N_B) = mean_term + cov_term (both >=0). With ridge=0 the split is exactly invariant
    to any global affine map (the paper's identifiability result); pass ridge>0 only for a mild
    conditioner on already-z-scored inputs. Requires n>d for a meaningful cov term (PCA-truncate)."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    mu0, mu1 = A.mean(0), B.mean(0)
    d = A.shape[1]
    S0 = np.cov(A, rowvar=False)
    S1 = np.cov(B, rowvar=False)
    if ridge:
        S0 = S0 + np.eye(d) * ridge
        S1 = S1 + np.eye(d) * ridge
    S1inv = np.linalg.pinv(S1)
    mean_term = float((mu1 - mu0) @ S1inv @ (mu1 - mu0))
    _, ld1 = np.linalg.slogdet(S1)
    _, ld0 = np.linalg.slogdet(S0)
    cov_term = float(np.trace(S1inv @ S0) - d + (ld1 - ld0))
    return 0.5 * (mean_term + cov_term), 0.5 * mean_term, 0.5 * cov_term


def mahalanobis_to_pool(this, pool, ridge=1e-3):
    """Mahalanobis distance of this subject's mean to the pool, in the pool's within covariance."""
    this = np.asarray(this, float)
    pool = np.asarray(pool, float)
    S = np.cov(pool, rowvar=False) + np.eye(pool.shape[1]) * ridge
    dmu = this.mean(0) - pool.mean(0)
    return float(math.sqrt(max(0.0, dmu @ np.linalg.pinv(S) @ dmu)))


def knn_distance_to_pool(this, pool, k=5, n=400, rng=None):
    """Mean distance from this subject's (subsampled) points to their k-NN in the pool."""
    rng = rng or np.random.default_rng(0)
    this = _subsample(this, n, rng)
    pool = _subsample(pool, 4 * n, rng)
    from sklearn.neighbors import NearestNeighbors
    kk = min(k, len(pool))
    if kk < 1 or len(this) < 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=kk).fit(pool)
    d, _ = nn.kneighbors(this)
    return float(d.mean())


def mmd_to_pool(X, subjects, seed=42, gamma="median", cap=800, min_n=20):
    """Per-subject MMD from each subject to the rest of the pool. Returns {subject: mmd}.

    Shape-agnostic: subjects with <min_n windows (or an empty pool) are skipped. Both sides capped
    to `cap` rows for cost; `mmd_rbf` subsamples further and clamps at 0.
    """
    X = np.asarray(X, float)
    subjects = np.asarray(subjects)
    rng = np.random.default_rng(seed)
    out = {}
    for s in np.unique(subjects):
        this = X[subjects == s]
        rest = X[subjects != s]
        if len(this) < min_n or len(rest) < min_n:
            continue
        if len(this) > cap:
            this = this[rng.choice(len(this), cap, replace=False)]
        if len(rest) > cap:
            rest = rest[rng.choice(len(rest), cap, replace=False)]
        out[int(s)] = mmd_rbf(this, rest, gamma=gamma, rng=rng)
    return out


def corr_across_subjects(stat_by_subj, acc_by_subj):
    """Pearson r between a per-subject statistic and per-subject accuracy, on shared subjects."""
    common_s = sorted(set(stat_by_subj) & set(acc_by_subj))
    if len(common_s) < 5:
        return dict(r=float("nan"), p=float("nan"), n=len(common_s))
    x = np.array([stat_by_subj[s] for s in common_s], float)
    y = np.array([acc_by_subj[s] for s in common_s], float)
    r, p, n = pearson(x, y)
    return dict(r=r, p=p, n=n)


def h_divergence(A, B, groups_a, groups_b, n=400, rng=None):
    """A-distance d_H = 2(1-2*err) of a group-vs-group RF classifier, folds grouped by TRIAL id so
    50%-overlapping windows never leak. Requires trial groups (else the estimate saturates)."""
    rng = rng or np.random.default_rng(0)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, GroupKFold
    ia = np.arange(len(A)) if len(A) <= n else rng.choice(len(A), n, replace=False)
    ib = np.arange(len(B)) if len(B) <= n else rng.choice(len(B), n, replace=False)
    X = np.vstack([np.asarray(A)[ia], np.asarray(B)[ib]])
    y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    ga = np.asarray(groups_a)[ia].astype(np.int64)
    gb = np.asarray(groups_b)[ib].astype(np.int64)
    gb = gb + (int(ga.max()) + 1 if ga.size else 0) + 1
    g = np.r_[ga, gb]
    n_splits = int(min(5, len(np.unique(g))))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return float("nan")
    err = 1 - cross_val_score(RandomForestClassifier(n_estimators=20, random_state=0),
                              X, y, cv=GroupKFold(n_splits=n_splits), groups=g).mean()
    return float(max(0.0, 2 * (1 - 2 * err)))


# =====================================================================================
# 6. Representation transforms
# =====================================================================================
def per_subject_center(X, subjects):
    Y = np.array(X, float, copy=True)
    for s in np.unique(subjects):
        m = subjects == s
        Y[m] -= Y[m].mean(0)
    return Y


def per_subject_zscore(X, subjects):
    Y = np.array(X, float, copy=True)
    for s in np.unique(subjects):
        m = subjects == s
        mu, sd = Y[m].mean(0), Y[m].std(0)
        Y[m] = (Y[m] - mu) / np.where(sd < EPS, 1.0, sd)
    return Y


def _sqrtm_psd(S, eps=1e-8, inverse=False):
    """Symmetric PSD (inverse) square root via eigendecomposition, eigenvalues floored at eps."""
    w, V = np.linalg.eigh((S + S.T) / 2.0)
    w = np.clip(w, eps, None)
    r = 1.0 / np.sqrt(w) if inverse else np.sqrt(w)
    return (V * r) @ V.T


def coral(Xs, Xt, ridge=1e-3):
    """CORAL: recolor centered source Xs to the target covariance. Aligns SECOND moments (means are
    handled separately). Returns Xs mapped so cov(out) ~ cov(Xt)."""
    Xs = np.asarray(Xs, float)
    Xt = np.asarray(Xt, float)
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + np.eye(d) * ridge
    Ct = np.cov(Xt, rowvar=False) + np.eye(d) * ridge
    W = _sqrtm_psd(Cs, inverse=True) @ _sqrtm_psd(Ct)   # whiten by Cs^-1/2, recolor by Ct^1/2
    return (Xs - Xs.mean(0)) @ W + Xs.mean(0)


def pca_embed(X_fit, X_apply, n_comp, whiten=True, seed=0):
    """Linear 'learned' embedding: PCA fit on X_fit (subject-agnostic), applied to X_apply."""
    from sklearn.decomposition import PCA
    n_comp = int(max(2, min(n_comp, X_fit.shape[1], X_fit.shape[0] - 1)))
    p = PCA(n_components=n_comp, whiten=whiten, random_state=seed).fit(X_fit)
    return p.transform(X_apply)


def rff_embed(X_fit, X_apply, n_comp=128, gamma="median", seed=0, rng=None):
    """Nonlinear 'learned' embedding via Random Fourier Features (approximates an RBF feature map).

    A torch-free stand-in for a learned nonlinear representation: it lets Module 3/5 be re-run in a
    DIFFERENT (nonlinear) space to test representation-robustness without a training loop.
    """
    rng = rng or np.random.default_rng(seed)
    X_fit = np.asarray(X_fit, float)
    X_apply = np.asarray(X_apply, float)
    d = X_fit.shape[1]
    if gamma == "median":
        g = median_gamma(X_fit, X_fit, rng)
    else:
        g = 1.0 / d if gamma == "1/d" else float(gamma)
    W = rng.standard_normal((d, n_comp)) * math.sqrt(2.0 * g)
    b = rng.uniform(0, 2 * math.pi, n_comp)
    return math.sqrt(2.0 / n_comp) * np.cos(X_apply @ W + b)


# =====================================================================================
# 7. Leakage-safe classifiers
# =====================================================================================
def _make_clf(name):
    if name == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis()
    if name == "svm":
        from sklearn.svm import LinearSVC
        return LinearSVC(C=1.0, dual="auto", max_iter=2000)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=1)
    raise ValueError(name)


def loso_accuracy(X, y, subjects, clf="lda", standardise=True, min_test=5):
    """Per-subject leave-one-subject-out accuracy. Train-only standardisation (no leakage).

    Returns {subject: accuracy}. Subjects with a single training class or <min_test test rows are
    skipped (graceful on tiny/odd datasets)."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    accs = {}
    for s in sorted(np.unique(subjects)):
        tr = subjects != s
        te = subjects == s
        if len(np.unique(y[tr])) < 2 or te.sum() < min_test:
            continue
        Xtr, Xte = X[tr], X[te]
        if standardise:
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        try:
            model = _make_clf(clf).fit(Xtr, y[tr])
            accs[int(s)] = float((model.predict(Xte) == y[te]).mean())
        except Exception:
            continue
    return accs


# =====================================================================================
# 8. Frame utilities
# =====================================================================================
def _cols_for(frame, features):
    cols = []
    for fb in features:
        cols += [c for c in frame.columns if c.startswith(fb + "_c")]
    return cols


def basis(frame, features=None):
    """Global-z-scored feature matrix for a feature family list (default REPR_BASIS)."""
    cols = _cols_for(frame, features or REPR_BASIS)
    X = np.nan_to_num(frame[cols].to_numpy(np.float64))
    return zscore(X, axis=0), cols


def n_channels(frame):
    try:
        return int(frame.attrs["n_channels"])
    except Exception:
        return len({c.split("_c")[-1] for c in frame.columns if c.startswith("MAV_c")})


def feature_layout(frame, features=None):
    """(X_raw, {channel: [col positions]}) over the given feature families (default REPR_BASIS)."""
    features = features or REPR_BASIS
    cols, chan_to_idx = [], {}
    for c in range(n_channels(frame)):
        idxs = []
        for fb in features:
            name = f"{fb}_c{c}"
            if name in frame.columns:
                idxs.append(len(cols))
                cols.append(name)
        if idxs:
            chan_to_idx[c] = idxs
    X_raw = np.nan_to_num(frame[cols].to_numpy(np.float64)) if cols else np.zeros((len(frame), 0))
    return X_raw, chan_to_idx


def resolve_jobs(n_jobs):
    if n_jobs in (None, 0, -1):
        return os.cpu_count() or 1
    return max(1, int(n_jobs))


def maybe_parallel(jobs, n_jobs):
    """jobs: list of (fn, args_tuple). Serial for n_jobs in (0,1,None); else joblib/loky."""
    if not jobs:
        return []
    if n_jobs in (None, 0, 1):
        return [fn(*args) for fn, args in jobs]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=resolve_jobs(n_jobs), backend="loky")(
        delayed(fn)(*args) for fn, args in jobs)


def build_frame(dataset, seed=42, normalize="global", decimate=1):
    """Lazy wrapper around the leakage-safe cached loader (imports h5/semg only here)."""
    from dsprofile import windows
    return windows.build_fast_frame(dataset, seed=seed, normalize=normalize, decimate=decimate)


# =====================================================================================
# 9. Dataset runner (atomic / resume / shard / error-isolated)
# =====================================================================================
def run_over_datasets(datasets, run_one, tag, seed=42, n_jobs=1, force=False):
    """Process each dataset with run_one(ds, seed, n_jobs) -> dict; one atomic per-dataset JSON in
    results/<tag>/. Resume-safe (skip existing unless force), error-isolated, shard-safe."""
    out_dir = results_dir(tag)
    out = {}
    for ds in datasets:
        p = out_dir / f"{ds}__{tag}.json"
        if p.exists() and not force:
            log(f"[SKIP] {tag} :: {ds}")
            try:
                out[ds] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                out[ds] = dict(note="existing file unreadable; use force=True")
            continue
        t0 = time.perf_counter()
        try:
            r = run_one(ds, seed, n_jobs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"[FAIL] {tag} :: {ds} :: {type(e).__name__}: {e}")
            out[ds] = dict(error=f"{type(e).__name__}: {e}")
            continue
        atomic_write_json(p, r)
        out[ds] = r
        log(f"[OK] {tag} :: {ds} :: {time.perf_counter() - t0:.1f}s")
    return out


# =====================================================================================
# 10. Synthetic ground-truth frame generator
# =====================================================================================
def synth_frame(kind="real_difficulty", n_subjects=16, n_channels=6, n_classes=12,
                per_class=40, sep=2.5, seed=0):
    """A frame with REPR_BASIS columns and a KNOWN structure, for ground-truth tests.

    kinds:
      'separable'        — clean class structure, subjects identical (control: high acc, ~0 shift).
      'shift_amplitude'  — subjects differ only by a global amplitude scale (shift lives in amplitude
                           features; a de-amplituded basis must see ~0 shift).
      'shift_shape'      — subjects differ by a covariance ROTATION only (shift survives de-amplituding;
                           CORAL must remove it, centring must not).
      'mean_shift'       — subjects differ by a per-subject MEAN offset (centring removes it).
      'real_difficulty'  — latent per-subject hardness lowers acc AND raises distance-to-pool,
                           independent of the global accuracy level (the money-result ground truth).
      'pure_floor'       — one shared distribution, near-random labels (no real per-subject signal).
    """
    rng = np.random.default_rng(seed)
    D = len(REPR_BASIS) * n_channels
    classM = rng.standard_normal((n_classes, D))
    # latent column j -> feature family REPR_BASIS[j // n_channels]; mark the amplitude-family columns
    # so a subject 'gain' scales EXACTLY the amplitude basis (and leaves the shape basis untouched).
    amp_dims = np.array([REPR_BASIS[j // n_channels] in AMPLITUDE_FEATURES for j in range(D)])
    feats, subj, lab, rep = [], [], [], []
    for s in range(n_subjects):
        gain = 1.0
        offset = np.zeros(D)
        rot = np.eye(D)
        csep = sep
        if kind == "shift_amplitude":
            gain = float(np.exp(rng.uniform(-0.7, 0.7)))
        elif kind == "shift_shape":
            A = rng.standard_normal((D, D)) * 0.15
            rot = np.eye(D) + (A - A.T)                       # near-orthogonal perturbation
        elif kind == "mean_shift":
            offset = rng.standard_normal(D) * 2.0
        elif kind == "real_difficulty":
            h = rng.uniform(0.0, 1.0)
            offset = rng.standard_normal(D)
            offset = offset / (np.linalg.norm(offset) + 1e-9) * (3.0 * h)
            csep = sep * (1.0 - 0.9 * h)
        elif kind == "pure_floor":
            csep = 0.05
        for c in range(n_classes):
            base = classM[c] * csep + rng.standard_normal((per_class, D))
            X = base @ rot.T
            X = X + offset
            g = np.where(amp_dims, gain, 1.0)                 # amplitude blocks scale, shape blocks don't
            X = X * g
            feats.append(X)
            subj += [s] * per_class
            lab += [c] * per_class
            rep += list(range(per_class))
    F = np.vstack(feats)
    cols, j = {}, 0
    for fb in REPR_BASIS:
        for ch in range(n_channels):
            cols[f"{fb}_c{ch}"] = F[:, j]
            j += 1
    df = pd.DataFrame(cols)
    df["subject"] = subj
    df["session"] = 0
    df["repetition"] = rep
    df["label"] = lab
    df.attrs["n_channels"] = n_channels
    df.attrs["dataset"] = f"synth_{kind}"
    return df
