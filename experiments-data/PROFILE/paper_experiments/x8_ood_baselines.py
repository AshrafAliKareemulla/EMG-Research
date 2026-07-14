"""X8 — Difficulty / OOD baseline bake-off. Buries "is MMD the best cheap statistic?" (§10).

Ranks MMD-to-pool against standard OOD scores (Mahalanobis-to-pool, kNN-mean-distance-to-pool, energy
distance) as per-subject difficulty predictors — per dataset (Pearson vs LOSO) and pooled across
datasets (leave-one-COHORT-out Spearman), so the "cheap MMD suffices" claim is earned, not asserted.

GROUND TRUTH: inject one subject from a clearly shifted distribution -> EVERY score must rank it top-1
most-OOD before any is trusted for ranking.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from . import common

SCORES = ["mmd_to_pool", "mahalanobis_to_pool", "knn_dist_to_pool", "energy_to_pool"]


def per_subject_scores(X, subjects, seed=42, cap=800):
    X = np.asarray(X, float)
    subjects = np.asarray(subjects)
    rng = np.random.default_rng(seed)
    out = {}
    for s in np.unique(subjects):
        this, rest = X[subjects == s], X[subjects != s]
        if len(this) < 20 or len(rest) < 20:
            continue
        t = this if len(this) <= cap else this[rng.choice(len(this), cap, replace=False)]
        r = rest if len(rest) <= cap else rest[rng.choice(len(rest), cap, replace=False)]
        out[int(s)] = dict(
            mmd_to_pool=common.mmd_rbf(t, r, rng=rng),
            mahalanobis_to_pool=common.mahalanobis_to_pool(t, r),
            knn_dist_to_pool=common.knn_distance_to_pool(t, r, rng=rng),
            energy_to_pool=common.energy_distance(t, r, rng=rng),
        )
    return out


def run_one(dataset, seed=42, n_jobs=1):
    from dsprofile import config
    with common.timer(f"X8 :: {dataset}"):
        frame = common.build_frame(dataset, seed=seed)
        X, _ = common.basis(frame)
        y, subj = frame.label.to_numpy(), frame.subject.to_numpy()
        acc = common.loso_accuracy(X, y, subj, "lda")
        scores = per_subject_scores(X, subj, seed)
    subs = sorted(set(acc) & set(scores))
    per_score = {}
    for sc in SCORES:
        x = np.array([scores[s][sc] for s in subs], float)
        a = np.array([acc[s] for s in subs], float)
        r, p, n = common.pearson(x, a)
        per_score[sc] = dict(pearson_r=r, p=p, n=n)
    return dict(dataset=dataset, cohort=config.COHORTS.get(dataset, dataset),
                n_subjects=len(subs), per_score=per_score,
                per_subject={int(s): dict(loso_acc=float(acc[s]), **scores[s]) for s in subs})


def build_pooled(tag="x8"):
    """Leave-one-COHORT-out Spearman of each score's predicted-vs-actual accuracy, pooled across datasets."""
    from dsprofile import config
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr
    rows = []
    for f in sorted(glob.glob(str(common.results_dir(tag) / f"*__{tag}.json"))):
        d = json.loads(Path(f).read_text())
        if "per_subject" not in d:
            continue
        for s, rec in d["per_subject"].items():
            rows.append(dict(dataset=d["dataset"], cohort=d.get("cohort", d["dataset"]),
                             subject=s, **rec))
    if len(rows) < 20:
        return dict(note="too few subjects pooled; run more datasets", n=len(rows))
    import pandas as pd
    A = pd.DataFrame(rows)
    for c in ["loso_acc"] + SCORES:
        A["z_" + c] = A.groupby("dataset")[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    result = {}
    for sc in SCORES:
        vals = []
        for coh in sorted(A.cohort.unique()):
            tr, te = A[A.cohort != coh], A[A.cohort == coh]
            if len(te) < 4 or len(tr) < 20 or te["loso_acc"].std() < 1e-9:
                continue
            m = Ridge(alpha=1.0).fit(tr[["z_" + sc]], tr["z_loso_acc"])
            r, _ = spearmanr(m.predict(te[["z_" + sc]]), te["loso_acc"])
            if r == r:
                vals.append(float(r))
        result[sc] = dict(lodo_mean_spearman=float(np.mean(vals)) if vals else float("nan"),
                          n_cohorts=len(vals))
    ranked = sorted(result, key=lambda k: -(result[k]["lodo_mean_spearman"]
                                            if result[k]["lodo_mean_spearman"] == result[k]["lodo_mean_spearman"] else -1))
    out = dict(n_subjects=len(rows), ranking=ranked, per_score=result,
               mmd_is_best_or_tied=bool(ranked and (ranked[0] == "mmd_to_pool" or
                    abs(result["mmd_to_pool"]["lodo_mean_spearman"] - result[ranked[0]]["lodo_mean_spearman"]) < 0.05)))
    common.atomic_write_json(common.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------ ground truth
def selftest(check):
    rng = np.random.default_rng(0)
    d = 5
    # 8 in-distribution subjects + 1 clearly shifted subject
    X, subj = [], []
    for s in range(8):
        X.append(rng.standard_normal((300, d)))
        subj += [s] * 300
    X.append(rng.standard_normal((300, d)) + 6.0)     # subject 8 is OOD
    subj += [8] * 300
    X = np.vstack(X); subj = np.array(subj)
    sc = per_subject_scores(X, subj, seed=0)
    for name in SCORES:
        vals = {s: sc[s][name] for s in sc}
        top = max(vals, key=lambda s: vals[s])
        check(f"X8 {name}: ranks the injected OOD subject top-1", top == 8, f"argmax={top}")
