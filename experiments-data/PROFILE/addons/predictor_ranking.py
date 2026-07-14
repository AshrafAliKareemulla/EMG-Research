"""Experiment D — which single cheap statistic best predicts subject difficulty?

The paper fixes MMD-to-pool as the PRIMARY predictor a priori (correctly, to avoid the
winner's curse of picking the best-of-4 per dataset). This experiment turns that choice into an
EMPIRICAL result: it ranks the four cheap per-subject statistics
    {mmd_to_pool, hdiv_to_pool, kl_mean_to_pool, kl_cov_to_pool}
by out-of-sample (leave-one-COHORT-out) predictive power, and tests whether adding a second
statistic to MMD beats MMD alone.

INPUT (no feature frames needed): results/module5/<ds>__difficulty.parquet, which already holds
per-subject {loso_acc + the 4 predictors} from the fresh module-5 run. Runs in seconds.

Every predictor is standardised WITHIN each dataset first (so a dataset's scale/baseline can't
drive the pooled result), exactly as the SDI does.

Output: results/experiments/exp_D_predictor_ranking.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from dsprofile import config
from dsprofile import stats as st

PREDICTORS = ["mmd_to_pool", "hdiv_to_pool", "kl_mean_to_pool", "kl_cov_to_pool"]


def _load():
    rows = []
    for f in config.find_all("module5", "*__difficulty.parquet"):
        ds = Path(f).name.split("__")[0]
        d = pd.read_parquet(f)
        missing = [c for c in ["loso_acc"] + PREDICTORS if c not in d.columns]
        if missing or len(d) < 5 or d["loso_acc"].std() < 1e-9:
            continue
        d = d.copy()
        d["dataset"] = ds
        d["cohort"] = config.COHORTS.get(ds, ds)
        for c in ["loso_acc"] + PREDICTORS:
            d["z_" + c] = (d[c] - d[c].mean()) / (d[c].std() + 1e-9)
        rows.append(d)
    if not rows:
        raise FileNotFoundError("no module5 difficulty parquets found; run module 5 first")
    return pd.concat(rows, ignore_index=True)


def _lodo_spearman(A, cols):
    """Leave-one-COHORT-out: fit z_loso ~ z(cols) on the other cohorts, predict the held-out
    cohort, and Spearman-correlate the prediction with actual LOSO accuracy. Mean over cohorts.
    Cohorts (not datasets) are held out so grabmyo_flow_static/dynamic etc. can't leak."""
    from sklearn.linear_model import LinearRegression
    zc = ["z_" + c for c in cols]
    vals = {}
    for coh in sorted(A.cohort.unique()):
        tr = A[A.cohort != coh]
        te = A[A.cohort == coh]
        if len(te) < 4 or len(tr) < 20 or te["loso_acc"].std() < 1e-9:
            continue
        m = LinearRegression().fit(tr[zc], tr["z_loso_acc"])
        pred = m.predict(te[zc])
        r, _ = spearmanr(pred, te["loso_acc"])
        if r == r:
            vals[coh] = float(r)
    return vals


def run():
    config.ensure_dirs()
    A = _load()
    out = dict(n_subjects=int(len(A)), n_datasets=int(A.dataset.nunique()),
               n_cohorts=int(A.cohort.nunique()), predictors={})

    for p in PREDICTORS:
        pr, pp = pearsonr(A["z_" + p], A["z_loso_acc"])
        perds, pmap = {}, {}
        for ds in sorted(A.dataset.unique()):
            sub = A[A.dataset == ds]
            if sub[p].std() < 1e-9 or sub["loso_acc"].std() < 1e-9:
                continue
            r, pv = pearsonr(sub[p].to_numpy(), sub["loso_acc"].to_numpy())
            perds[ds] = dict(r=float(r), p=float(pv), n=int(len(sub)))
            pmap[ds] = pv
        fdr = st.fdr_dict(pmap)
        for ds in perds:
            perds[ds]["q_fdr"] = fdr[ds]["q"]
            perds[ds]["sig_fdr"] = fdr[ds]["significant_fdr"]
        n_sig = sum(1 for ds, v in perds.items() if v["sig_fdr"] and v["r"] < 0)
        lodo = _lodo_spearman(A, [p])
        out["predictors"][p] = dict(
            pooled_pearson_r=float(pr), pooled_p=float(pp),
            n_datasets_sig_correct_sign=int(n_sig),
            n_datasets_wrong_sign=int(sum(1 for v in perds.values() if v["r"] > 0)),
            lodo_mean_spearman=float(np.mean(list(lodo.values()))) if lodo else float("nan"),
            lodo_median_spearman=float(np.median(list(lodo.values()))) if lodo else float("nan"),
            lodo_n_cohorts=len(lodo),
            per_dataset=perds,
        )

    def _lodo(p):
        v = out["predictors"][p]["lodo_mean_spearman"]
        return v if v == v else -1.0

    rank = sorted(out["predictors"], key=lambda p: -_lodo(p))
    out["ranking_by_lodo"] = rank

    # Tie handling: LODO Spearman over 9 cohorts has a standard error of roughly ~0.1, so
    # differences under TIE_EPS are noise. If MMD is within TIE_EPS of the top, we keep MMD as
    # the recommended statistic (it is the a-priori primary AND the canonical marginal-shift
    # metric), and record the tie honestly rather than crowning a 0.001 winner.
    TIE_EPS = 0.02
    top = _lodo(rank[0])
    tied = [p for p in rank if top - _lodo(p) <= TIE_EPS]
    out["top_lodo_spearman"] = float(top)
    out["tied_for_best"] = tied
    out["tie_eps"] = TIE_EPS
    if "mmd_to_pool" in tied:
        out["best_single_predictor"] = "mmd_to_pool"
        out["best_is_tie"] = len(tied) > 1
    else:
        out["best_single_predictor"] = rank[0]
        out["best_is_tie"] = False

    # Does adding a second statistic to MMD beat MMD alone, OUT-OF-SAMPLE?
    base = _lodo_spearman(A, ["mmd_to_pool"])
    base_mean = float(np.mean(list(base.values()))) if base else float("nan")
    combos = {"mmd_to_pool (alone)": base_mean}
    for p in PREDICTORS:
        if p == "mmd_to_pool":
            continue
        v = _lodo_spearman(A, ["mmd_to_pool", p])
        combos[f"mmd_to_pool + {p}"] = float(np.mean(list(v.values()))) if v else float("nan")
    allfour = _lodo_spearman(A, PREDICTORS)
    combos["all four"] = float(np.mean(list(allfour.values()))) if allfour else float("nan")
    out["combination_lodo_mean_spearman"] = combos
    best_combo = max(combos, key=lambda k: combos[k] if combos[k] == combos[k] else -1)
    out["mmd_alone_is_best_or_tied"] = bool(
        combos["mmd_to_pool (alone)"] >= max(v for v in combos.values() if v == v) - 0.02)
    out["best_combination"] = best_combo
    tie_note = (f"MMD is tied for best (within {TIE_EPS}) with {[p for p in tied if p != 'mmd_to_pool']}"
                if out["best_is_tie"] else f"{rank[0]} is the clear best")
    out["verdict"] = (
        f"Single-predictor ranking by LODO Spearman: {tie_note}. "
        + ("MMD alone is best or within 0.02 of the best COMBINATION -> report MMD as THE cheap "
           "statistic; combining predictors does not help out-of-sample (consistent with the "
           "negative combined_cv_r2 in module5)."
           if out["mmd_alone_is_best_or_tied"] else
           f"Best combination is '{best_combo}' -> combining helps; report it."))

    outdir = config.RESULTS_DIR / "experiments"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "exp_D_predictor_ranking.json").write_text(json.dumps(out, indent=2))
    print("\n=== D: predictor ranking (leave-one-cohort-out) ===")
    for p in rank:
        v = out["predictors"][p]
        print(f"  {p:18} pooled r={v['pooled_pearson_r']:+.3f}  "
              f"LODO rho={v['lodo_mean_spearman']:+.3f}  "
              f"sig(FDR,correct-sign)={v['n_datasets_sig_correct_sign']}/{out['n_datasets']}  "
              f"wrong-sign={v['n_datasets_wrong_sign']}")
    print("\n  combinations (LODO mean Spearman):")
    for k, v in out["combination_lodo_mean_spearman"].items():
        print(f"    {k:28} {v:+.3f}")
    print("\n  " + out["verdict"])
    return out


if __name__ == "__main__":
    run()
