"""SDI — sEMG Subject-Difficulty Index (Phase-2, our proposed novel metric).

A single training-free number that predicts a subject's cross-subject (LOSO) difficulty from
cheap distribution-shift statistics, PORTABLE across datasets. Because raw predictor scales and
LOSO baselines differ wildly between datasets, everything is standardised WITHIN each dataset;
the SDI then predicts a subject's *relative* difficulty within its dataset.

  SDI(subject) = w . z(predictors)         (higher SDI = harder = lower LOSO accuracy)

Weights `w` are fit once (pooled, within-dataset-standardised) and VALIDATED leave-one-dataset-out
(LODO): train on 13 datasets, predict the held-out dataset's subject ranking. This is the
contribution none of the reviewed papers do — a cross-dataset difficulty index for sEMG ADL.

Aggregates the per-dataset outputs of Module 5 (results/module5/*__difficulty.parquet); run AFTER
Module 5 has completed on all datasets. No feature frames needed.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PREDICTORS = ["mmd_to_pool", "hdiv_to_pool", "kl_mean_to_pool", "kl_cov_to_pool"]


def _load_standardised():
    """Load every dataset's per-subject difficulty parquet; z-score predictors + target
    WITHIN each dataset. Returns the stacked frame with a 'dataset' column."""
    rows = []
    for f in config.find_all("module5", "*__difficulty.parquet"):
        ds = Path(f).name.split("__")[0]
        d = pd.read_parquet(f)
        if len(d) < 4 or d["loso_acc"].std() < 1e-9:
            continue
        z = pd.DataFrame({"dataset": ds, "subject": d["subject"].values,
                          "loso_acc": d["loso_acc"].values})
        for c in ["loso_acc"] + PREDICTORS:
            z["z_" + c] = (d[c].values - d[c].mean()) / (d[c].std() + 1e-9)
        rows.append(z)
    if not rows:
        raise FileNotFoundError("no Module-5 difficulty parquets found; run --module 5 first")
    return pd.concat(rows, ignore_index=True)


def _fit(A):
    """Fit z_loso_acc ~ w . z(predictors) (Ridge, pooled). SDI = -(prediction)."""
    from sklearn.linear_model import Ridge
    X = A[["z_" + p for p in PREDICTORS]].to_numpy()
    y = A["z_loso_acc"].to_numpy()
    m = Ridge(alpha=1.0).fit(X, y)
    return m


def _score(A):
    from scipy.stats import spearmanr, pearsonr
    from sklearn.linear_model import Ridge
    datasets = sorted(A["dataset"].unique())

    # Leave-one-COHORT-out, not leave-one-dataset-out. Holding out grabmyo_flow_static while
    # grabmyo_flow_dynamic (same 20 subjects) stays in the training set is not a held-out
    # dataset; it leaks the cohort. `lodo` is keyed by dataset but the split removes the whole
    # cohort, so the reported generalisation is to genuinely unseen subjects.
    lodo = {}
    for ds in datasets:
        coh = config.COHORTS.get(ds, ds)
        held = [d for d in datasets if config.COHORTS.get(d, d) == coh]
        tr = A[~A.dataset.isin(held)]; te = A[A.dataset == ds]
        if len(te) < 4 or len(tr) < 20:
            continue
        m = Ridge(alpha=1.0).fit(tr[["z_" + p for p in PREDICTORS]], tr["z_loso_acc"])
        pred_acc = m.predict(te[["z_" + p for p in PREDICTORS]])
        # SDI = -pred_acc (difficulty). Correlate predicted accuracy with ACTUAL accuracy.
        r, p = spearmanr(pred_acc, te["loso_acc"])
        lodo[ds] = dict(spearman=float(r), p_value=float(p), n=int(len(te)),
                        cohort=coh, cohort_held_out=held)

    # pooled fit + in-sample R^2 (for the weights we report)
    m_all = _fit(A)
    r2 = float(m_all.score(A[["z_" + p for p in PREDICTORS]].to_numpy(), A["z_loso_acc"].to_numpy()))
    weights = {p: float(w) for p, w in zip(PREDICTORS, m_all.coef_)}

    # SDI vs best single predictor (does combining beat the best one?) — pooled
    single = {}
    for p in PREDICTORS:
        rr, pp = pearsonr(A["z_" + p], A["z_loso_acc"])
        single[p] = dict(pearson_r=float(rr), p_value=float(pp))
    return weights, r2, lodo, single


def run():
    config.ensure_dirs()
    from . import stats as st
    A = _load_standardised()
    weights, r2, lodo, single = _score(A)
    valid = [v["spearman"] for v in lodo.values()]

    fdr = st.fdr_dict({d: v["p_value"] for d, v in lodo.items()})
    for d, v in fdr.items():
        lodo[d]["q_fdr"] = v["q"]
        lodo[d]["significant_fdr"] = v["significant_fdr"]

    clean = {d: v for d, v in lodo.items() if d not in config.OUTLIER_DATASETS}
    cv = [v["spearman"] for v in clean.values()]

    result = dict(
        n_subjects=int(len(A)), n_datasets=int(A["dataset"].nunique()),
        n_independent_cohorts=len({config.COHORTS.get(d, d) for d in A["dataset"].unique()}),
        weights=weights,
        pooled_r2=r2,
        pooled_r2_note="IN-SAMPLE ridge fit on all subjects; not a generalisation estimate. "
                       "Use lodo_* (leave-one-cohort-out) for that.",
        validation="leave-one-COHORT-out (whole cohort removed from training)",
        lodo_mean_spearman=float(np.mean(valid)) if valid else float("nan"),
        lodo_median_spearman=float(np.median(valid)) if valid else float("nan"),
        lodo_mean_spearman_without_outliers=float(np.mean(cv)) if cv else float("nan"),
        lodo_median_spearman_without_outliers=float(np.median(cv)) if cv else float("nan"),
        outliers_excluded=list(config.OUTLIER_DATASETS),
        n_datasets_significant_fdr=int(sum(v.get("significant_fdr", False) for v in lodo.values())),
        n_datasets_negative_spearman=int(sum(v["spearman"] < 0 for v in lodo.values())),
        lodo_per_dataset=lodo,
        single_predictor_pooled=single,
        honest_summary=(
            "The SDI transfers to an unseen cohort only weakly. Report lodo_mean_spearman with "
            "the per-dataset FDR-corrected table, not the pooled in-sample R^2, and state that "
            "the index is reliable on high-n datasets and near-chance on the n=10 ones."),
    )
    outdir = config.RESULTS_DIR / "module6_sdi"
    outdir.mkdir(parents=True, exist_ok=True)
    A.to_parquet(outdir / "sdi_standardised_table.parquet")
    (outdir / "sdi.json").write_text(json.dumps(result, indent=2))
    return result
