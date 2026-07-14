"""Assemble the cross-dataset comparison matrix from per-module JSON outputs.

Rebuilt 2026-07-10 against the corrected result keys:
  * `knn_loo_acc` (shuffled 5-fold over overlapping windows) -> `knn_trial_cv_acc` +
    `knn_loso_acc`, so the within- vs cross-subject gap is visible in the headline table.
  * `difficulty_r2` was `combined_linear_r2`, an IN-SAMPLE fit -> now `combined_cv_r2`.
  * `difficulty_best_predictor` was post-hoc best-of-4 -> now the a-priori primary predictor,
    with the post-hoc pick kept in a clearly-named column.
  * `kl_mean_over_cov` came from the invariance-broken E3 -> replaced by the null-corrected
    `mean_share_of_excess` from block_c.
  * `A4_intersubj_over_interday` now reads block_c's fair (same-granularity) comparison.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config


def _load(module, dataset, kind):
    p = config.find(module, f"{dataset}__{kind}.json")   # live tree first, then frozen
    return json.loads(p.read_text()) if p.exists() else {}


def _get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def build(datasets=None):
    datasets = datasets or config.ALL14
    rows = []
    for ds in datasets:
        m2 = _load("module2", ds, "separability")
        m3 = _load("module3", ds, "shift")
        m4 = _load("module4", ds, "channels")
        m5 = _load("module5", ds, "difficulty")
        bc = _load("block_c", ds, "block_c")
        bd = _load("block_d", ds, "block_d")
        rb = _load("robust_difficulty", ds, "robust_difficulty")
        ac = _load("actionability", ds, "actionability")

        e2 = bc.get("E2_a4_fair") or {}
        pooled = _get(bc, "E3_meancov", "representations", "pooled", default={}) or {}
        e6 = bd.get("E6_sampling_rate") or {}

        rows.append(dict(
            dataset=ds,
            cohort=config.COHORTS.get(ds, ds),
            n_subjects=m5.get("n_subjects"),
            n_classes=m2.get("n_classes"),
            n_channels=m2.get("n_channels") or m4.get("n_channels"),
            # --- separability (both protocols; the gap is the story) -------------------
            fisher_ratio=m2.get("fisher_ratio"),
            silhouette=m2.get("silhouette"),
            knn_trial_cv_acc=m2.get("knn_trial_cv_acc"),
            knn_loso_acc=m2.get("knn_loso_acc"),
            within_minus_cross=m2.get("within_minus_cross"),
            pca95_dim=m2.get("pca95_dim"),
            twonn_dim=m2.get("twonn_dim"),
            # --- shift -----------------------------------------------------------------
            inter_subject_mmd=_get(m3, "inter_subject", "mmd_frob"),
            mean_share_of_excess=pooled.get("mean_share_of_excess"),
            shift_detectable=pooled.get("shift_detectable"),
            kl_removed_by_subject_center=_get(bc, "E3_meancov",
                                              "kl_excess_removed_by_subject_center"),
            A4_applicable=e2.get("applicable"),
            A4_intersubj_over_interday=e2.get("ratio_inter_subject_over_inter_day"),
            A4_p_value=e2.get("p_value"),
            # --- channels / fs ---------------------------------------------------------
            mean_channel_nmi=m4.get("mean_nmi"),
            min_channels_90pct=m4.get("min_channels_for_90pct_relevance"),
            min_channels_95pct_loso=_get(bd, "E7_channel_reduction",
                                         "min_channels_for_95pct_loso"),
            fs_sufficiency_testable=e6.get("testable"),
            native_fs=e6.get("native_fs"),
            # --- difficulty ------------------------------------------------------------
            loso_acc_mean=m5.get("loso_acc_mean"),
            difficulty_primary_predictor=m5.get("primary_predictor"),
            difficulty_primary_r=m5.get("primary_pearson_r"),
            difficulty_primary_p=m5.get("primary_p_value"),
            difficulty_cv_r2=m5.get("combined_cv_r2"),
            difficulty_insample_r2=m5.get("combined_insample_r2"),
            difficulty_best_predictor_posthoc=m5.get("best_predictor_posthoc"),
            classifiers_agree=rb.get("classifiers_agree"),
            # --- actionability (report the ceiling next to the advantage) --------------
            sdi_guided_advantage=ac.get("guided_advantage"),
            sdi_oracle_ceiling=ac.get("oracle_ceiling"),
            sdi_guided_significant=ac.get("guided_beats_random_significantly"),
            mmd_vs_calibration_gain_r=_get(ac, "mmd_vs_calibration_gain", "pearson_r"),
        ))
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "cross_dataset_matrix.xlsx"
    try:
        df.to_excel(out, index=False)
    except Exception:
        out = config.RESULTS_DIR / "cross_dataset_matrix.csv"
        df.to_csv(out, index=False)
    return df, str(out)
