"""Block F — "Science of sEMG datasets": dataset-level meta-characterization.

Treats the 14 datasets as data points. Runs LOCALLY on the Phase-1 artifacts (per-module JSONs,
module3 shift-matrix npz, module5 difficulty parquets) — no feature frames needed. Answers:
  * what makes a dataset hard? (dataset stats vs mean cross-subject accuracy)
  * a statistical atlas of sEMG datasets (PCA + clustering of dataset fingerprints)
  * meta-analysis: pooled effect size of "distribution shift predicts difficulty" (random-effects)
  * how many subjects are 'enough'? (shift-estimate stability vs #subjects, from the npz matrices)
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, windows


def _j(m, ds, suf):
    p = config.find(m, f"{ds}__{suf}.json")
    return json.loads(p.read_text()) if p.exists() else {}


def dataset_table():
    """One row per dataset: its full statistical fingerprint + mean LOSO accuracy + difficulty r."""
    dsets = sorted(Path(x).name.split("__")[0]
                   for x in config.find_all("module2", "*__separability.json"))
    rows = []
    for ds in dsets:
        m2 = _j("module2", ds, "separability"); m3 = _j("module3", ds, "shift")
        m4 = _j("module4", ds, "channels"); m5 = _j("module5", ds, "difficulty")
        m1 = _j("module1", ds, "card")
        iss = m3.get("inter_subject", {})
        pc = m5.get("predictor_correlations", {}).get(config.PRIMARY_PREDICTOR, {})
        rows.append(dict(
            dataset=ds, cohort=config.COHORTS.get(ds, ds),
            n_subjects=m5.get("n_subjects"), n_classes=m2.get("n_classes"),
            n_channels=m2.get("n_channels"),
            fisher=m2.get("fisher_ratio"), silhouette=m2.get("silhouette"),
            # leakage-safe replacements for the old, misnamed `knn_loo_acc`
            knn_trial_cv=m2.get("knn_trial_cv_acc"), knn_loso=m2.get("knn_loso_acc"),
            pca95=m2.get("pca95_dim"), twonn=m2.get("twonn_dim"),
            is_mmd=iss.get("mmd_frob"), is_hdiv=iss.get("hdiv_frob"),
            is_klmean=iss.get("kl_mean_frob"), is_klcov=iss.get("kl_cov_frob"),
            mean_nmi=m4.get("mean_nmi"),
            loso_acc=m5.get("loso_acc_mean"),
            diff_r_mmd=pc.get("pearson_r"),
            entropy_valid=bool(windows.entropy_valid(ds)),
            sampen=(m1.get("complexity_median") or {}).get("SAMPEN"),
            hfd=(m1.get("complexity_median") or {}).get("HFD"),
        ))
    return pd.DataFrame(rows)


# Predictors that are themselves CLASSIFIER ACCURACIES measured on the same features as the
# target. Correlating them with mean LOSO accuracy is close to tautological ("accuracy predicts
# accuracy") and tells you nothing about what property of the DATA makes a dataset hard. The
# old headline — knn_loo vs loso_acc, rho = 0.93 — was exactly this, computed from a leaky
# 5-fold at that. They are still reported, but flagged and excluded from the ranking.
CIRCULAR = {"knn_trial_cv", "knn_loso"}

# Complexity summaries are undefined on datasets whose 250 ms window holds < 200 samples.
ENTROPY_DEPENDENT = {"sampen", "hfd"}

PREDICTORS = ["n_classes", "n_channels", "fisher", "silhouette", "knn_trial_cv", "knn_loso",
              "pca95", "twonn", "is_mmd", "is_hdiv", "is_klmean", "is_klcov", "mean_nmi",
              "sampen", "hfd"]


def what_makes_hard(T, alpha=0.05):
    """Correlate each dataset-level statistic with mean cross-subject accuracy (n=14 datasets).

    Corrections over the original:
      * Benjamini-Hochberg FDR across the predictors (POST_RESULTS_PLAN Stage 1 required it;
        nothing implemented it, and 14 uncorrected tests on n=14 points is a lot of rope).
      * accuracy-valued predictors are marked `circular` and excluded from the headline ranking.
      * entropy-derived predictors are restricted to the datasets where entropy is defined.
    """
    from scipy.stats import pearsonr, spearmanr
    from . import stats as st
    y = T["loso_acc"].to_numpy(dtype=float)
    out, sp_p = {}, {}
    for c in PREDICTORS:
        if c not in T.columns:
            continue
        x = T[c].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if c in ENTROPY_DEPENDENT and "entropy_valid" in T.columns:
            ok &= T["entropy_valid"].to_numpy(bool)
        if ok.sum() < 5 or np.nanstd(x[ok]) < 1e-12:
            continue
        pr, pp = pearsonr(x[ok], y[ok]); sr, spv = spearmanr(x[ok], y[ok])
        out[c] = dict(pearson_r=float(pr), pearson_p=float(pp),
                      spearman_r=float(sr), spearman_p=float(spv),
                      n_datasets=int(ok.sum()), circular=bool(c in CIRCULAR))
        sp_p[c] = spv

    # FDR over the NON-circular predictors only (the circular ones are not hypotheses).
    honest = {c: p for c, p in sp_p.items() if c not in CIRCULAR}
    fdr = st.fdr_dict(honest, alpha=alpha)
    for c, v in fdr.items():
        out[c]["spearman_q_fdr"] = v["q"]
        out[c]["significant_fdr"] = v["significant_fdr"]
    for c in out:
        if c in CIRCULAR:
            out[c]["spearman_q_fdr"] = None
            out[c]["significant_fdr"] = None
            out[c]["note"] = ("classifier accuracy vs classifier accuracy — near-tautological; "
                              "excluded from FDR and from the ranking")

    ranked = dict(sorted(out.items(),
                         key=lambda kv: (kv[1]["circular"], -abs(kv[1]["spearman_r"]))))
    survivors = [c for c, v in out.items() if v.get("significant_fdr")]
    return dict(per_predictor=ranked, fdr_alpha=alpha,
                survivors_after_fdr=survivors,
                headline=("no dataset-level property survives FDR" if not survivors
                          else f"survives FDR: {', '.join(survivors)}"))


def atlas(T):
    """PCA(2) + hierarchical clustering of the standardised dataset fingerprints."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering
    cols = ["n_classes", "n_channels", "fisher", "silhouette", "knn_trial_cv", "twonn",
            "is_mmd", "is_hdiv", "is_klmean", "is_klcov", "mean_nmi"]
    cols = [c for c in cols if c in T.columns]
    X = T[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=np.nanmean(X))
    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2).fit(Xs)
    coords = pcs.transform(Xs)
    k = min(4, len(T) - 1)
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
    return dict(
        coords={T.dataset.iloc[i]: [float(coords[i, 0]), float(coords[i, 1])] for i in range(len(T))},
        cluster={T.dataset.iloc[i]: int(labels[i]) for i in range(len(T))},
        explained_variance=[float(v) for v in pcs.explained_variance_ratio_],
        loadings={c: [float(pcs.components_[0][j]), float(pcs.components_[1][j])]
                  for j, c in enumerate(cols)},
    )


def pool_random_effects(rs, ns):
    """DerSimonian-Laird random-effects meta-analysis of correlations via the Fisher-z transform.
    Pure function (no I/O) so it is unit-testable. rs: correlations, ns: sample sizes (>=4).
    All arithmetic in float64; arctanh is clipped off +-1 to avoid infinities."""
    rs = np.asarray(rs, dtype=np.float64)
    ns = np.asarray(ns, dtype=np.float64)
    z = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    w = ns - 3.0                                   # inverse-variance weights for Fisher-z
    sw = np.sum(w)
    z_fixed = np.sum(w * z) / sw
    Q = float(np.sum(w * (z - z_fixed) ** 2))
    df = len(z) - 1
    C = sw - np.sum(w ** 2) / sw
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_re = 1.0 / (1.0 / w + tau2)
    z_re = np.sum(w_re * z) / np.sum(w_re)
    se_re = np.sqrt(1.0 / np.sum(w_re))
    return dict(
        k_datasets=int(len(rs)), pooled_r_random_effects=float(np.tanh(z_re)),
        ci95=[float(np.tanh(z_re - 1.96 * se_re)), float(np.tanh(z_re + 1.96 * se_re))],
        tau2=float(tau2), Q=Q, df=int(df),
        I2=float(max(0.0, (Q - df) / Q) if Q > 0 else 0.0),
        pooled_r_fixed=float(np.tanh(z_fixed)),
    )


def _collect(predictor, min_subjects):
    rows = []
    for f in config.find_all("module5", "*__difficulty.json"):
        d = json.loads(open(f).read())
        ds = d.get("dataset") or Path(f).name.split("__")[0]
        pc = d.get("predictor_correlations", {}).get(predictor)
        if pc and d.get("n_subjects", 0) >= min_subjects:
            rows.append(dict(dataset=ds, cohort=config.COHORTS.get(ds, ds),
                             r=pc["pearson_r"], n=d["n_subjects"], p=pc["p_value"]))
    return rows


def meta_analysis(predictor=None, min_subjects=5, alpha=0.05):
    """Pool each dataset's <predictor>-vs-LOSO-accuracy correlation across datasets.

    `predictor` defaults to config.PRIMARY_PREDICTOR (fixed a priori — pooling the per-dataset
    *best-of-4* would pool a winner's curse).

    Three additions over the original:
      * per-dataset FDR across the k datasets, so "significant on 9/14" is stated honestly.
      * a leave-one-COHORT-out pooled estimate. The 14 datasets are not 14 independent samples:
        grabmyo_flow_static and grabmyo_flow_dynamic are one cohort, the four EMAHA sets are one,
        ninapro_db4/db5 are one. Treating k=14 as independent overstates the evidence and
        deflates tau^2.
      * a without-outlier estimate (senic reverses sign; I^2 ~ 0.79 says the effects are
        genuinely heterogeneous, so a single pooled r is a weak summary either way).
    """
    from . import stats as st
    predictor = predictor or config.PRIMARY_PREDICTOR
    rows = _collect(predictor, min_subjects)
    if len(rows) < 2:
        return dict(k_datasets=len(rows), note="too few datasets to pool")

    base = pool_random_effects([r["r"] for r in rows], [r["n"] for r in rows])
    base["predictor"] = predictor

    fdr = st.fdr_dict({r["dataset"]: r["p"] for r in rows}, alpha=alpha)
    base["per_dataset"] = {r["dataset"]: dict(r=r["r"], n=r["n"], p=r["p"],
                                              q_fdr=fdr[r["dataset"]]["q"],
                                              significant_fdr=fdr[r["dataset"]]["significant_fdr"],
                                              sign_as_expected=bool(r["r"] < 0))
                           for r in rows}
    base["n_significant_fdr"] = int(sum(v["significant_fdr"] for v in base["per_dataset"].values()))
    base["n_significant_fdr_correct_sign"] = int(sum(
        v["significant_fdr"] and v["sign_as_expected"] for v in base["per_dataset"].values()))
    base["n_wrong_sign"] = int(sum(not v["sign_as_expected"] for v in base["per_dataset"].values()))

    # one dataset per cohort: keep the largest-n member of each cohort
    by_cohort = {}
    for r in rows:
        if r["cohort"] not in by_cohort or r["n"] > by_cohort[r["cohort"]]["n"]:
            by_cohort[r["cohort"]] = r
    ind = list(by_cohort.values())
    base["n_independent_cohorts"] = len(ind)
    if len(ind) >= 2:
        base["pooled_one_per_cohort"] = pool_random_effects([r["r"] for r in ind],
                                                            [r["n"] for r in ind])
        base["pooled_one_per_cohort"]["datasets_used"] = [r["dataset"] for r in ind]

    keep = [r for r in rows if r["dataset"] not in config.OUTLIER_DATASETS]
    if len(keep) >= 2:
        base["pooled_without_outliers"] = pool_random_effects([r["r"] for r in keep],
                                                              [r["n"] for r in keep])
        base["pooled_without_outliers"]["excluded"] = list(config.OUTLIER_DATASETS)

    base["heterogeneity_warning"] = (
        f"I^2 = {base['I2']:.2f}. Effects range from {min(r['r'] for r in rows):+.2f} to "
        f"{max(r['r'] for r in rows):+.2f}; a single pooled r summarises a genuinely "
        f"heterogeneous set and should be reported with the forest plot, not alone.")
    return base


def how_many_subjects(reps=30, seed=0):
    """Stability of the inter-subject MMD estimate vs #subjects: subsample the MMD matrix (from the
    Phase-1 npz) to k subjects, recompute the Frobenius aggregate, report mean +/- std vs k."""
    rng = np.random.default_rng(seed)
    out = {}
    for f in config.find_all("module3", "*__shift_matrices.npz"):
        ds = Path(f).name.split("__")[0]
        M = np.load(f)["mmd"]
        n = M.shape[0]
        if n < 12:
            continue
        curve = {}
        for k in range(4, n + 1, 2):
            vals = []
            for _ in range(reps):
                idx = rng.choice(n, k, replace=False)
                sub = M[np.ix_(idx, idx)]
                off = sub[~np.eye(k, dtype=bool)]
                vals.append(off.mean())
            curve[k] = [float(np.mean(vals)), float(np.std(vals))]
        out[ds] = dict(full_n=n, curve=curve)
    return out


def run():
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "meta"
    outdir.mkdir(parents=True, exist_ok=True)
    T = dataset_table()
    T.to_parquet(outdir / "dataset_table.parquet")
    result = dict(
        n_datasets=int(len(T)),
        n_independent_cohorts=int(T["cohort"].nunique()) if "cohort" in T.columns else None,
        cohort_map={r.dataset: r.cohort for r in T.itertuples()} if "cohort" in T.columns else None,
        what_makes_hard=what_makes_hard(T),
        atlas=atlas(T),
        meta_analysis=meta_analysis(),
        how_many_subjects=how_many_subjects(),
        caveats=[
            "The 14 datasets span fewer independent cohorts (see cohort_map); k=14 overstates "
            "the evidence in any pooled statistic.",
            "The atlas mixes device, fs and channel count with population differences.",
            "Dataset-level correlations use n=14 points; FDR-corrected q values are reported.",
        ],
    )
    (outdir / "meta.json").write_text(json.dumps(result, indent=2))
    return result
