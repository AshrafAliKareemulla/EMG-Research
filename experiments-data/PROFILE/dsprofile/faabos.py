"""E9 — FAABOS ADL-taxonomy profiling (Block E / ADL angle).

Re-runs the separability + difficulty analyses on EMAHA's coarse FAABOS activity grouping
(the hierarchical ADL taxonomy) instead of the fine gesture labels. Answers whether coarse ADL
categories are more separable / more cross-subject-stable than fine gestures (scientific Q I2).

Only applies to datasets whose manifest has a `faabos_group` column (emaha_db1). Scalable: it looks
the column up dynamically and no-ops for datasets without it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, windows
from . import cv
from .module2_separability import fisher_ratio, mahalanobis_si
from .module3_shift import _basis, mmd_rbf, h_divergence
from .module5_difficulty import loso_lda_accuracy, subject_shift_stats


def _remap_to_faabos(frame, dataset):
    """Map each window's fine label -> FAABOS group using the manifest (label -> faabos_group)."""
    mpath = config.L1_ROOT / dataset / "manifest.parquet"
    if not mpath.exists():
        return None
    m = pd.read_parquet(mpath)
    if "faabos_group" not in m.columns:
        return None
    mapping = (m[["label", "faabos_group"]].drop_duplicates()
               .set_index("label")["faabos_group"].to_dict())
    fa = frame["label"].map(mapping)
    out = frame.copy()
    out["label"] = fa.to_numpy()
    return out[out["label"].notna()].reset_index(drop=True)


def run(dataset, seed=42):
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "faabos"
    outdir.mkdir(parents=True, exist_ok=True)
    frame = windows.build_fast_frame(dataset, seed=seed)
    fa = _remap_to_faabos(frame, dataset)
    if fa is None:
        result = dict(dataset=dataset, note="no faabos_group column; skipped")
        (outdir / f"{dataset}__faabos.json").write_text(json.dumps(result, indent=2))
        return result
    X = _basis(fa); y = fa["label"].astype("category").cat.codes.to_numpy()
    subjects = fa["subject"].to_numpy()
    target = loso_lda_accuracy(X, y, subjects)
    preds = subject_shift_stats(X, subjects, seed, trials=cv.trial_ids(fa))
    common = sorted(set(target) & set(preds))
    corr = None
    if len(common) >= 4:
        from scipy.stats import pearsonr
        acc = np.array([target[s] for s in common])
        mmd = np.array([preds[s]["mmd_to_pool"] for s in common])
        r, p = pearsonr(mmd, acc)
        corr = dict(pearson_r=float(r), p_value=float(p))
    groups = cv.trial_ids(fa)
    result = dict(
        dataset=dataset, n_faabos_classes=int(fa["label"].nunique()),
        n_subjects=len(target),
        fisher_ratio=fisher_ratio(X, y),
        # both honest protocols (the old `knn_loo_acc` was a shuffled 5-fold over
        # 50%-overlapping windows, i.e. leaked); FAABOS is the paper's only ADL-specific
        # result, so its numbers must not be the inflated ones.
        knn_trial_cv_acc=cv.knn_trial_cv(X, y, groups, seed=seed),
        knn_loso_acc=cv.knn_loso(X, y, subjects, seed=seed),
        mahalanobis_si=mahalanobis_si(X, y),
        loso_acc_mean=float(np.mean(list(target.values()))) if target else None,
        chance_level=float(1.0 / max(1, fa["label"].nunique())),
        difficulty_corr_mmd=corr,
    )
    (outdir / f"{dataset}__faabos.json").write_text(json.dumps(result, indent=2))
    return result
