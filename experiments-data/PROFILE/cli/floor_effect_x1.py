"""Experiment X1 — the floor-effect confound, done correctly.

WHY THIS EXISTS
---------------
The money result is "cheap MMD-to-pool predicts each subject's cross-subject (LOSO) difficulty."
But that correlation is strongest exactly where accuracy is near the floor (ninapro_db1 acc 0.12 ->
r=-0.77, ninapro_db2 acc 0.14 -> r=-0.73) and vanishes where accuracy is healthy (grabmyo acc 0.70 ->
r=+0.03). A reviewer will ask: are we predicting DIFFICULTY, or DISTANCE-FROM-THE-ACCURACY-FLOOR?

The old `floor_effect.py` cannot answer it: it reports a single per-dataset trend, its committed
`floor_effect.json` self-contradicts (grabmyo +0.95 "floor" vs ninapro_db2 -0.93 "reverse"), and the
"two-term ceiling+variance" narrative pools ~29 NESTED channel-count rungs (k=1 subset of k=2 subset
of ...) as if independent and quotes p<0.001 — the same pseudo-replication fixed elsewhere (F3/F4).

X1 replaces that with THREE clean probes (all reuse the cached frames; no new data):

  X1a  MATCH THE CLASS COUNT.   Sub-sample the high-class datasets down to grabmyo's ~17 classes and
       refit r, over many random class subsets. r stays strongly negative -> not a class-count/floor
       artifact; r collapses -> it rode the class-count floor.

  X1b  MATCH THE ACCURACY.       Cripple a healthy dataset (drop channels via the mRMR order) until its
       LOSO accuracy lands near the floor (~0.15), refit r. r turns strongly negative -> the predictor
       only works once the model is already failing (a real, reportable limitation); r stays ~0 -> the
       healthy-dataset null is intrinsic, not a floor artifact.

  X1c  DO THE POOLED MODEL CORRECTLY.  Build rungs from DISJOINT RANDOM channel subsets (not nested
       prefixes, so rungs are exchangeable), then estimate whether |r| depends on mean accuracy
       (ceiling effect) and/or on accuracy spread (variance effect) with a DATASET-CLUSTERED BOOTSTRAP
       (the dataset is the resampling unit, so within-dataset rung correlation cannot inflate
       significance). The naive pseudo-replicated correlation is reported alongside, clearly labelled,
       only to show the contrast. Optional statsmodels MixedLM if installed.

GROUND TRUTH (validate the code before trusting real output: `python floor_effect_x1.py --selftest`)
  * PURE-FLOOR synthetic (all subjects one distribution, near-random labels): X1 must NOT report a
    stable real effect.
  * REAL-DIFFICULTY synthetic (a latent per-subject hardness that lowers accuracy AND raises MMD,
    INDEPENDENT of the global accuracy level): X1a must recover a floor-invariant negative r.

EVERY OUTCOME IS PUBLISHABLE. Predictor survives -> the headline is real, report it. Predictor is
floor-only -> "cheap statistics predict who a model fails on, but only once it is already failing" — a
genuine, scoped contribution. Either way, delete the unsupported "RESOLVED" label.

RUN (on Ubuntu)
  python floor_effect_x1.py --selftest                 # ~1 min, validates the code on synthetic GT
  python floor_effect_x1.py --datasets grabmyo,ninapro_db1,ninapro_db2,ninapro_db4,ninapro_db5,\
      fors_emg,emaha_db1,senic --jobs 8                 # the real run
Output: results/floor_effect_x1/<dataset>__x1.json  +  results/floor_effect_x1/pooled.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from dsprofile import config, progress  # noqa: E402

# ---- decision thresholds (documented, tweakable) ------------------------------------
STRONG_NEG = -0.30     # a "strong-ish" negative correlation
NEAR_ZERO = 0.20       # |r| below this ~ no signal
DEFAULT_TARGET_CLASSES = 17    # grabmyo's class count (the reference)
DEFAULT_TARGET_ACC = 0.15      # the ninapro-family floor to cripple a healthy dataset down to


# =====================================================================================
# Local re-implementations (kept here so this file imports WITHOUT h5py/semg, i.e. the
# pure functions + synthetic ground truth are testable on any box). These are byte-for-byte
# equivalent to dsprofile.module3_shift.mmd_rbf and module5_difficulty.loso_lda_accuracy.
# =====================================================================================
def _mmd_rbf(A, B, gamma=None, n=400, rng=None):
    rng = rng or np.random.default_rng(0)
    A = A if len(A) <= n else A[rng.choice(len(A), n, replace=False)]
    B = B if len(B) <= n else B[rng.choice(len(B), n, replace=False)]
    if gamma is None:
        gamma = 1.0 / A.shape[1]
    from sklearn.metrics.pairwise import rbf_kernel
    Kaa = rbf_kernel(A, A, gamma).mean()
    Kbb = rbf_kernel(B, B, gamma).mean()
    Kab = rbf_kernel(A, B, gamma).mean()
    return float(max(0.0, Kaa + Kbb - 2 * Kab))


def _loso_lda_accuracy(X, y, subjects):
    """Per-subject LOSO accuracy with LDA, train-fold-only standardisation."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    accs = {}
    for s in sorted(np.unique(subjects)):
        tr = subjects != s
        te = subjects == s
        if len(np.unique(y[tr])) < 2 or te.sum() < 5:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LinearDiscriminantAnalysis().fit((X[tr] - mu) / sd, y[tr])
        accs[int(s)] = float((clf.predict((X[te] - mu) / sd) == y[te]).mean())
    return accs


def _standardise(X):
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.where(sd < 1e-12, 1.0, sd)


def _n_channels(frame):
    try:
        return int(frame.attrs["n_channels"])
    except Exception:
        chs = {c.split("_c")[-1] for c in frame.columns if c.startswith("MAV_c")}
        return len(chs)


def _feature_layout(frame):
    """Return (X_raw (n, D), {channel: [column positions]}) for the REPR_BASIS columns.

    A 'channel' contributes all len(REPR_BASIS) features, exactly as the real pipeline treats it.
    """
    cols, chan_to_idx = [], {}
    for c in range(_n_channels(frame)):
        idxs = []
        for fb in config.REPR_BASIS:
            name = f"{fb}_c{c}"
            if name in frame.columns:
                idxs.append(len(cols))
                cols.append(name)
        if idxs:
            chan_to_idx[c] = idxs
    X_raw = np.nan_to_num(frame[cols].to_numpy(np.float64))
    return X_raw, chan_to_idx


def _mmd_to_pool(X, subjects, seed, cap=800):
    """Per-subject mean MMD to the rest of the pool, on whatever basis X is."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in np.unique(subjects):
        this = X[subjects == s]
        rest = X[subjects != s]
        if len(this) < 20 or len(rest) < 20:
            continue
        if len(this) > cap:
            this = this[rng.choice(len(this), cap, replace=False)]
        if len(rest) > cap:
            rest = rest[rng.choice(len(rest), cap, replace=False)]
        out[int(s)] = _mmd_rbf(this, rest, rng=rng)
    return out


def _r_core(X, y, subj, seed):
    """The load-bearing measurement: Pearson r between per-subject MMD-to-pool and LOSO accuracy.

    Returns dict(mean_acc, acc_std, r, abs_r, n) or None if too few usable subjects.
    """
    acc = _loso_lda_accuracy(X, y, subj)
    mmd = _mmd_to_pool(X, subj, seed)
    common = sorted(set(acc) & set(mmd))
    if len(common) < 5:
        return None
    a = np.array([acc[s] for s in common], float)
    m = np.array([mmd[s] for s in common], float)
    if a.std() < 1e-9 or m.std() < 1e-9:
        return None
    r = float(np.corrcoef(m, a)[0, 1])
    return dict(mean_acc=float(a.mean()), acc_std=float(a.std()), r=r, abs_r=abs(r), n=len(common))


def _rung_metrics(X_raw, chan_to_idx, chans, y, subj, seed):
    """One rung = the r measured on a channel subset. Standardises the subset before measuring."""
    idx = [i for c in chans for i in chan_to_idx[c]]
    if not idx:
        return None
    d = _r_core(_standardise(X_raw[:, idx]), y, subj, seed)
    if d is None:
        return None
    d["k"] = len(chans)
    return d


# =====================================================================================
# Pure statistics (unit-tested against known ground truth)
# =====================================================================================
def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z (Frisch-Waugh-Lovell residualisation)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_bootstrap_partial(table, target, control, B=2000, seed=0, cluster_key="dataset"):
    """Cluster-bootstrap of partial_corr(abs_r, <target> | <control>).

    The CLUSTER (default 'dataset'; pass 'cohort' for the honest independent unit — grabmyo_flow
    static/dynamic and ninapro_db4/db5 each share a cohort) is the resampling unit, so correlated
    rungs within a cluster cannot inflate significance. `table` is a list of dicts with 'abs_r',
    <target>/<control>, and the cluster_key (falls back to 'dataset' if the key is absent).
    """
    rng = np.random.default_rng(seed)
    key = cluster_key if all(cluster_key in r for r in table) else "dataset"
    clusters = sorted({r[key] for r in table})
    by = {c: [r for r in table if r[key] == c] for c in clusters}
    point = partial_corr([r["abs_r"] for r in table],
                         [r[target] for r in table], [r[control] for r in table])
    vals = []
    for _ in range(B):
        samp = rng.choice(clusters, len(clusters), replace=True)
        rows = [r for c in samp for r in by[c]]
        v = partial_corr([r["abs_r"] for r in rows],
                         [r[target] for r in rows], [r[control] for r in rows])
        if v == v:
            vals.append(v)
    lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals \
        else (float("nan"), float("nan"))
    return dict(partial_r=point, ci95=[lo, hi], n_rungs=len(table), cluster_key=key,
                n_datasets=len({r["dataset"] for r in table}), n_clusters=len(clusters),
                n_boot=len(vals), excludes_zero=bool(vals and (lo > 0 or hi < 0)))


def floor_verdict(ceiling, variance):
    """Turn the two clustered-bootstrap partials into an honest four-way verdict.

    ceiling  = partial(|r|, mean_acc | acc_std)  — NEGATIVE means |r| falls as accuracy rises (a
               ceiling / floor effect: the predictor fades near the accuracy ceiling).
    variance = partial(|r|, acc_std | mean_acc)  — POSITIVE means |r| rises with accuracy spread.
    """
    ceiling_present = bool(ceiling["excludes_zero"] and ceiling["partial_r"] < 0)
    variance_present = bool(variance["excludes_zero"] and variance["partial_r"] > 0)
    if ceiling_present and variance_present:
        head = ("TWO-TERM: a ceiling effect (|r| falls as accuracy rises) AND a variance effect "
                "(|r| rises with accuracy spread) are BOTH present, with dataset-clustered CIs "
                "excluding 0. This is the STATE.md §5 model — now actually supported.")
    elif ceiling_present:
        head = ("CEILING-ONLY: |r| falls as accuracy rises (a floor effect); no separable variance "
                "effect survives clustering. The predictor is real but weakens near the ceiling.")
    elif variance_present:
        head = ("VARIANCE-ONLY: |r| rises with accuracy spread; no separable ceiling effect. The "
                "predictor needs difficulty variance, not a low accuracy level.")
    else:
        head = ("NEITHER: no ceiling or variance effect survives dataset-clustered CIs — the floor "
                "'confound' is NOT supported at the between-rung level once nesting is removed.")
    return dict(headline=head, ceiling_present=ceiling_present, variance_present=variance_present)


# =====================================================================================
# The three probes
# =====================================================================================
def x1a_matched_class_count(frame, target_classes=DEFAULT_TARGET_CLASSES, n_subsets=20,
                            seed=42, n_jobs=1):
    """X1a — does the negative r survive when the task is reduced to `target_classes` classes?"""
    y = frame.label.to_numpy()
    subj = frame.subject.to_numpy()
    classes = np.unique(y)
    if len(classes) <= target_classes:
        return dict(applicable=False, n_classes=int(len(classes)), target_classes=int(target_classes),
                    note=f"n_classes <= target ({target_classes}); this dataset is a reference, not a test")
    X_raw, chan_to_idx = _feature_layout(frame)
    Xall = _standardise(X_raw)
    full = _r_core(Xall, y, subj, seed)

    rng = np.random.default_rng(seed)
    subsets = [rng.choice(classes, target_classes, replace=False) for _ in range(n_subsets)]

    def job(i, sc):
        mask = np.isin(y, sc)
        return _r_core(Xall[mask], y[mask], subj[mask], seed + 1 + i)

    res = _maybe_parallel([(job, (i, sc)) for i, sc in enumerate(subsets)], n_jobs)
    rr = np.array([d["r"] for d in res if d], float)
    accs = np.array([d["mean_acc"] for d in res if d], float)
    if full is None or rr.size < 3:
        return dict(applicable=False, note="too few usable subjects at full or matched class count")

    r_full = full["r"]
    r_m = float(rr.mean())
    ci = [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))]
    survives = None
    if r_m < STRONG_NEG and abs(r_m) >= 0.6 * abs(r_full):
        decision = ("SURVIVES — the negative r persists at matched class count; NOT a class-count/"
                    "floor artifact.")
        survives = True
    elif abs(r_m) < 0.5 * abs(r_full) or r_m > STRONG_NEG:
        decision = ("COLLAPSES — r weakens sharply once class count is matched; it rode the "
                    "class-count floor.")
        survives = False
    else:
        decision = "AMBIGUOUS — partial attenuation; report the numbers, do not over-claim."
    return dict(applicable=True, n_classes=int(len(classes)), target_classes=int(target_classes),
                r_full=float(r_full), acc_full=float(full["mean_acc"]),
                r_matched_mean=r_m, r_matched_ci95=ci, acc_matched_mean=float(accs.mean()),
                n_subsets=int(rr.size), survives_class_match=survives, decision=decision)


def x1b_accuracy_matched(frame, target_acc=DEFAULT_TARGET_ACC, seed=42, n_jobs=1):
    """X1b — cripple a healthy dataset to ~target_acc via the mRMR channel order; does r emerge?"""
    X_raw, chan_to_idx = _feature_layout(frame)
    y = frame.label.to_numpy()
    subj = frame.subject.to_numpy()
    try:
        order, C = _channel_order(frame)
    except Exception as e:
        return dict(applicable=False, note=f"channel order unavailable: {type(e).__name__}: {e}")
    ks = sorted(set([k for k in (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28) if k <= C] + [C]))

    def job(k):
        return _rung_metrics(X_raw, chan_to_idx, order[:k], y, subj, seed)

    curve = [d for d in _maybe_parallel([(job, (k,)) for k in ks], n_jobs) if d]
    if not curve:
        return dict(applicable=False, note="no usable rungs")
    full = max(curve, key=lambda d: d["k"])
    r_full, acc_full = full["r"], full["mean_acc"]
    if acc_full < target_acc + 0.05:
        return dict(applicable=False, acc_full=float(acc_full), target_acc=float(target_acc),
                    note="dataset is already at/near the floor; cannot be crippled further to test")
    cand = min(curve, key=lambda d: abs(d["mean_acc"] - target_acc))
    floor_dependent = None
    if abs(r_full) < NEAR_ZERO and cand["r"] < -0.40:
        decision = ("FLOOR-DEPENDENT — r is ~0 at full (healthy) accuracy but strongly negative once "
                    "the dataset is crippled to the floor: the predictor fires only when the model is "
                    "already failing.")
        floor_dependent = True
    elif r_full < STRONG_NEG:
        decision = "NOT floor-only — the predictor already works at full (healthy) accuracy."
        floor_dependent = False
    else:
        decision = "AMBIGUOUS/weak — report the curve."
    return dict(applicable=True, n_channels=int(C), target_acc=float(target_acc),
                r_full=float(r_full), acc_full=float(acc_full),
                r_at_target=float(cand["r"]), acc_at_target=float(cand["mean_acc"]),
                k_at_target=int(cand["k"]), floor_dependent=floor_dependent, decision=decision,
                curve=[{k2: d[k2] for k2 in ("k", "mean_acc", "acc_std", "r", "n")} for d in curve])


def x1c_rungs(frame, n_subsets=6, seed=42, n_jobs=1):
    """X1c — rungs from DISJOINT RANDOM channel subsets (exchangeable, not nested prefixes)."""
    X_raw, chan_to_idx = _feature_layout(frame)
    y = frame.label.to_numpy()
    subj = frame.subject.to_numpy()
    n_ch = len(chan_to_idx)
    chans_all = list(chan_to_idx)
    sizes = sorted(set([s for s in (2, 3, 4, 6, 8, 12, 16, 24) if s < n_ch] + [n_ch]))
    rng = np.random.default_rng(seed)
    specs = []
    for s in sizes:
        draws = 1 if s >= n_ch else n_subsets       # full set: only one possible subset
        for d in range(draws):
            specs.append((list(rng.choice(chans_all, s, replace=False)), seed + 1000 * s + d))

    def job(chans, sd):
        return _rung_metrics(X_raw, chan_to_idx, chans, y, subj, sd)

    dsname = frame.attrs.get("dataset", "?")
    cohort = config.COHORTS.get(dsname, dsname)
    out = []
    for d in _maybe_parallel([(job, (c, sd)) for c, sd in specs], n_jobs):
        if d:
            out.append(dict(dataset=dsname, cohort=cohort, k=d["k"], mean_acc=d["mean_acc"],
                            acc_std=d["acc_std"], r=d["r"], abs_r=d["abs_r"], n=d["n"]))
    return out


def build_pooled(all_rungs, per_dataset, bootstrap=2000, seed=0):
    """X1c pooled analysis + aggregation of the X1a/X1b per-dataset verdicts."""
    from scipy.stats import pearsonr
    result = dict(n_rungs=len(all_rungs),
                  n_datasets=len({r["dataset"] for r in all_rungs}))
    if len(all_rungs) >= 8 and result["n_datasets"] >= 3:
        # NAIVE, pseudo-replicated — reported ONLY for contrast; must not be quoted
        ar = [r["abs_r"] for r in all_rungs]
        ma = [r["mean_acc"] for r in all_rungs]
        nr, npv = pearsonr(ar, ma)
        result["naive_pseudoreplicated_DO_NOT_QUOTE"] = dict(
            pearson_abs_r_vs_mean_acc=float(nr), p_value=float(npv),
            warning="rungs within a dataset are correlated; this p is pseudo-replicated (the F3/F4 "
                    "error). Use the dataset-clustered bootstrap below.")
        # PRIMARY inference: COHORT-clustered (the honest independent unit — db4/db5 etc. share a cohort)
        ceiling = cluster_bootstrap_partial(all_rungs, "mean_acc", "acc_std", B=bootstrap, seed=seed,
                                            cluster_key="cohort")
        variance = cluster_bootstrap_partial(all_rungs, "acc_std", "mean_acc", B=bootstrap, seed=seed + 1,
                                             cluster_key="cohort")
        result["ceiling_effect_partial_absr_vs_meanacc"] = ceiling
        result["variance_effect_partial_absr_vs_accstd"] = variance
        result["ceiling_dataset_clustered_secondary"] = cluster_bootstrap_partial(
            all_rungs, "mean_acc", "acc_std", B=bootstrap, seed=seed, cluster_key="dataset")
        result["verdict"] = floor_verdict(ceiling, variance)
        result["caveats"] = dict(
            variance_term=("the +variance partial is partly RANGE-RESTRICTION (|r| is mechanically "
                           "attenuated when the outcome spread is small); treat the ceiling term as "
                           "PRIMARY and the variance term as secondary."),
            clusters=(f"primary CIs are COHORT-clustered (n_clusters={ceiling.get('n_clusters')}); a "
                      "small number of clusters makes the bootstrap CI itself uncertain -> report as "
                      "solid evidence, not proof. Run all 14 datasets for more clusters."),
            target=("difficulty here is the self-LDA-LOSO proxy; running X1 on a DL target (X3) would "
                    "make this floor resolution model-agnostic."))
        result["mixedlm"] = _mixedlm(all_rungs)
    else:
        result["note"] = "too few rungs/datasets for the pooled model; run more datasets"

    # aggregate the per-dataset probes
    a_survive = [ds for ds, v in per_dataset.items()
                 if (v.get("x1a") or {}).get("survives_class_match") is True]
    a_collapse = [ds for ds, v in per_dataset.items()
                  if (v.get("x1a") or {}).get("survives_class_match") is False]
    b_floor = [ds for ds, v in per_dataset.items()
               if (v.get("x1b") or {}).get("floor_dependent") is True]
    b_notfloor = [ds for ds, v in per_dataset.items()
                  if (v.get("x1b") or {}).get("floor_dependent") is False]
    result["x1a_survives_class_match"] = a_survive
    result["x1a_collapses"] = a_collapse
    result["x1b_floor_dependent"] = b_floor
    result["x1b_works_at_full_accuracy"] = b_notfloor
    result["overall_reading"] = (
        "Synthesise: (1) X1c verdict above (ceiling / variance / two-term / neither, clustered CIs); "
        "(2) X1a — datasets where the negative r SURVIVES a matched class count are evidence the "
        "predictor is not a class-count artifact; (3) X1b — datasets that become FLOOR-DEPENDENT when "
        "crippled are evidence the predictor needs a failing model. Report the honest combination; "
        "replace STATE.md §5's unsupported 'RESOLVED' with whatever this shows.")
    return result


def _mixedlm(all_rungs):
    """Optional statsmodels mixed-effects fit: abs_r ~ mean_acc + acc_std, random intercept/dataset."""
    try:
        import statsmodels.formula.api as smf
        df = pd.DataFrame(all_rungs)
        m = smf.mixedlm("abs_r ~ mean_acc + acc_std", df, groups=df["dataset"]).fit(reml=False)
        return dict(available=True,
                    params={k: float(v) for k, v in m.params.items()},
                    pvalues={k: float(v) for k, v in m.pvalues.items()},
                    note="MixedLM p-values are model-based (parametric); the clustered bootstrap above "
                         "is the assumption-lighter primary.")
    except Exception as e:
        return dict(available=False, note=f"statsmodels unavailable or fit failed: {type(e).__name__}: {e}")


def _channel_order(frame):
    """mRMR channel order (reuse Block D's ranking) so X1b degrades along the realistic path."""
    from dsprofile.module4_channels import (per_channel_fisher, nmi_matrix, greedy_mrmr,
                                            _channel_signal_matrix)
    M, _ = _channel_signal_matrix(frame, "RMS")
    relevance = per_channel_fisher(frame)
    nmi = nmi_matrix(M)
    return [int(c) for c in greedy_mrmr(relevance, nmi)], M.shape[1]


def _maybe_parallel(jobs, n_jobs):
    """jobs: list of (fn, args). Serial if n_jobs in (0,1,None); else joblib/loky."""
    if not jobs:
        return []
    if n_jobs in (None, 0, 1):
        return [fn(*args) for fn, args in jobs]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(*args) for fn, args in jobs)


# =====================================================================================
# Synthetic ground truth
# =====================================================================================
def synth_frame(kind="real_difficulty", n_subjects=16, n_channels=6, n_classes=24,
                per_class=40, sep=2.5, seed=0):
    """Build a frame with REPR_BASIS feature columns and a KNOWN floor structure.

    real_difficulty: a latent per-subject hardness h_s (uniform, independent of any global accuracy
                     level) both lowers class separability (-> lower LOSO acc) and adds a per-subject
                     offset (-> higher MMD-to-pool). So r(mmd, acc) is negative and floor-invariant.
    pure_floor:      one shared distribution, near-zero class separation, no per-subject offset ->
                     LOSO ~ chance, MMD ~ 0, r ~ noise.
    """
    rng = np.random.default_rng(seed)
    D = len(config.REPR_BASIS) * n_channels
    classM = rng.standard_normal((n_classes, D))
    feats, subj, lab, rep = [], [], [], []
    for s in range(n_subjects):
        if kind == "real_difficulty":
            h = rng.uniform(0.0, 1.0)
            off = rng.standard_normal(D)
            off = off / (np.linalg.norm(off) + 1e-9) * (3.0 * h)   # bigger offset -> higher MMD
            csep = sep * (1.0 - 0.9 * h)                            # more hardness -> lower acc
        else:  # pure_floor
            off = np.zeros(D)
            csep = 0.05
        for c in range(n_classes):
            mu = classM[c] * csep + off
            feats.append(mu + rng.standard_normal((per_class, D)))
            subj += [s] * per_class
            lab += [c] * per_class
            rep += list(range(per_class))
    F = np.vstack(feats)
    cols, j = {}, 0
    for fb in config.REPR_BASIS:
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


def selftest():
    """Validate the code against synthetic ground truth. Prints PASS/FAIL; returns True iff all pass."""
    ok = [True]

    def check(name, cond, detail=""):
        ok[0] &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}   {detail}")

    # --- pure partial_corr ---
    rng = np.random.default_rng(0)
    z = rng.standard_normal(600)
    x = z + 0.1 * rng.standard_normal(600)
    y = z + 0.1 * rng.standard_normal(600)
    pc = partial_corr(x, y, z)
    check("partial_corr: shared-only-via-z -> ~0", abs(pc) < 0.2, f"{pc:.3f}")
    w = rng.standard_normal(600)
    x2 = w + z + 0.1 * rng.standard_normal(600)
    y2 = w + z + 0.1 * rng.standard_normal(600)
    pc2 = partial_corr(x2, y2, z)
    check("partial_corr: shared-w-beyond-z -> >0.5", pc2 > 0.5, f"{pc2:.3f}")

    # --- floor_verdict on synthetic rung tables ---
    def rung_table(mode, seed=0):
        r = np.random.default_rng(seed)
        rows = []
        for d in range(9):
            for acc in np.linspace(0.10, 0.70, 8):
                spread = 0.05 + 0.01 * r.standard_normal()
                if mode == "floor":
                    absr = np.clip(0.85 - 1.0 * acc + 0.03 * r.standard_normal(), 0, 1)
                else:
                    absr = np.clip(0.40 + 0.03 * r.standard_normal(), 0, 1)
                rows.append(dict(dataset=f"d{d}", abs_r=float(absr),
                                 mean_acc=float(acc), acc_std=float(spread)))
        return rows

    cf = cluster_bootstrap_partial(rung_table("floor"), "mean_acc", "acc_std", B=400, seed=0)
    check("floor table: ceiling partial < 0 and CI excludes 0",
          cf["partial_r"] < 0 and cf["excludes_zero"], f"r={cf['partial_r']:.3f} ci={cf['ci95']}")
    cc = cluster_bootstrap_partial(rung_table("const"), "mean_acc", "acc_std", B=400, seed=0)
    check("constant table: ceiling partial CI includes 0",
          not cc["excludes_zero"], f"r={cc['partial_r']:.3f} ci={cc['ci95']}")

    # --- X1a on synthetic frames (thresholds set wide of the sampling noise so a CORRECT
    #     implementation passes deterministically; the SCIENTIFIC contrast is real<<0 vs floor~0) ---
    fr = synth_frame("real_difficulty", n_subjects=20, seed=1)
    a = x1a_matched_class_count(fr, target_classes=8, n_subsets=6, seed=1)
    check("X1a real-difficulty: r_full strongly negative", a.get("r_full", 0.0) < -0.4,
          f"r_full={a.get('r_full')}")
    check("X1a real-difficulty: negative r PERSISTS at matched class count",
          a.get("r_matched_mean", 0.0) < STRONG_NEG,
          f"r_matched={a.get('r_matched_mean')}  decision={a.get('decision','')[:32]}")
    ff = synth_frame("pure_floor", n_subjects=24, seed=1)
    b = x1a_matched_class_count(ff, target_classes=8, n_subsets=6, seed=1)
    check("X1a pure-floor: NOT a strong-negative r (no real difficulty signal)",
          b.get("r_full", 0.0) > -0.35, f"r_full={b.get('r_full')}")

    print(f"\n==== selftest: {'ALL PASS' if ok[0] else 'FAILURES ABOVE'} ====")
    return ok[0]


# =====================================================================================
# Driver
# =====================================================================================
def _atomic_write(path, obj):
    d = os.path.dirname(str(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)


def run_dataset(dataset, target_classes, target_acc, n_subsets, seed, n_jobs):
    from dsprofile import windows                       # lazy: needs semg/h5py + the dataset
    with progress.timer(f"X1 :: {dataset}"):
        frame = windows.build_fast_frame(dataset, seed=seed)
        out = dict(
            dataset=dataset,
            x1a=x1a_matched_class_count(frame, target_classes, n_subsets, seed, n_jobs),
            x1b=x1b_accuracy_matched(frame, target_acc, seed, n_jobs),
        )
        rungs = x1c_rungs(frame, n_subsets=max(4, n_subsets // 3), seed=seed, n_jobs=n_jobs)
    out["x1c_n_rungs"] = len(rungs)
    a = out["x1a"]
    b = out["x1b"]
    progress.log(f"  {dataset}: X1a={a.get('decision', a.get('note', ''))[:60]} | "
                 f"X1b={b.get('decision', b.get('note', ''))[:60]}")
    return out, rungs


def main():
    ap = argparse.ArgumentParser(description="Experiment X1 — floor-effect, done correctly")
    ap.add_argument("--datasets",
                    default="grabmyo,ninapro_db1,ninapro_db2,ninapro_db4,ninapro_db5,"
                            "fors_emg,emaha_db1,senic")
    ap.add_argument("--target-classes", type=int, default=DEFAULT_TARGET_CLASSES)
    ap.add_argument("--target-acc", type=float, default=DEFAULT_TARGET_ACC)
    ap.add_argument("--n-subsets", type=int, default=20, help="random class/channel subsets per probe")
    ap.add_argument("--bootstrap", type=int, default=2000, help="dataset-clustered bootstrap reps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--selftest", action="store_true", help="validate on synthetic ground truth and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    outdir = config.RESULTS_DIR / "floor_effect_x1"
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = (list(config.ALL14) if args.datasets.strip() == "all"
                else [d.strip() for d in args.datasets.split(",") if d.strip()])

    per_dataset, all_rungs = {}, []
    for ds in datasets:
        try:
            out, rungs = run_dataset(ds, args.target_classes, args.target_acc,
                                     args.n_subsets, args.seed, args.jobs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] X1 :: {ds} :: {type(e).__name__}: {e}", flush=True)
            per_dataset[ds] = dict(error=f"{type(e).__name__}: {e}")
            continue
        _atomic_write(outdir / f"{ds}__x1.json", out)
        per_dataset[ds] = out
        all_rungs += rungs
        print(f"[OK] X1 :: {ds}", flush=True)

    pooled = build_pooled(all_rungs, per_dataset, bootstrap=args.bootstrap, seed=args.seed)
    pooled["datasets"] = datasets
    _atomic_write(outdir / "pooled.json", pooled)

    print("\n=== X1 POOLED VERDICT (floor effect, done correctly) ===")
    if "verdict" in pooled:
        print("  " + pooled["verdict"]["headline"])
        print(f"  ceiling partial |r|~mean_acc : r={pooled['ceiling_effect_partial_absr_vs_meanacc']['partial_r']:+.3f} "
              f"ci={pooled['ceiling_effect_partial_absr_vs_meanacc']['ci95']}")
        print(f"  variance partial |r|~acc_std : r={pooled['variance_effect_partial_absr_vs_accstd']['partial_r']:+.3f} "
              f"ci={pooled['variance_effect_partial_absr_vs_accstd']['ci95']}")
    print(f"  X1a survives class-match : {pooled.get('x1a_survives_class_match')}")
    print(f"  X1a collapses            : {pooled.get('x1a_collapses')}")
    print(f"  X1b floor-dependent      : {pooled.get('x1b_floor_dependent')}")
    print(f"  X1b works at full acc    : {pooled.get('x1b_works_at_full_accuracy')}")
    print(f"\n  -> {outdir}/pooled.json + per-dataset files")


if __name__ == "__main__":
    main()
