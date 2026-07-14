"""Experiment E — permutation test for the per-subject difficulty correlations.

At n = 10-40 subjects, with senic reversed, a parametric Pearson p-value invites skepticism
(it assumes bivariate normality). This replaces it with an EXACT, assumption-light null: shuffle
the LOSO-accuracy labels across subjects B times and count how often the shuffled |correlation|
matches or beats the observed one. The only assumption is exchangeability of subjects under the
null, which is exactly what "MMD carries no information about difficulty" means.

Done for the PRIMARY predictor (MMD-to-pool) and reported beside the parametric p, with
Benjamini-Hochberg FDR across datasets.

INPUT (no feature frames): results/module5/<ds>__difficulty.parquet. Runs in <1 min.
Output: results/experiments/exp_E_permutation.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from dsprofile import config
from dsprofile import stats as st

PREDICTOR = config.PRIMARY_PREDICTOR      # "mmd_to_pool"
B = 20000                                  # permutations (exact enough for p ~ 1e-4)


def _perm_p(x, y, B=B, seed=0):
    """Two-sided permutation p for Pearson(x, y): P(|r_perm| >= |r_obs|). Add-one smoothed."""
    rng = np.random.default_rng(seed)
    obs = pearsonr(x, y)[0]
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum()) + 1e-30
    aobs = abs(float((xc * yc).sum() / denom))
    cnt = 0
    n = len(y)
    for _ in range(B):
        yp = yc[rng.permutation(n)]
        rp = abs(float((xc * yp).sum() / denom))
        if rp >= aobs - 1e-12:
            cnt += 1
    return float(obs), float((cnt + 1) / (B + 1))


def run():
    config.ensure_dirs()
    rows = {}
    for f in config.find_all("module5", "*__difficulty.parquet"):
        ds = Path(f).name.split("__")[0]
        d = pd.read_parquet(f)
        if PREDICTOR not in d.columns or "loso_acc" not in d.columns:
            continue
        if len(d) < 5 or d[PREDICTOR].std() < 1e-9 or d["loso_acc"].std() < 1e-9:
            continue
        x = d[PREDICTOR].to_numpy(float)
        y = d["loso_acc"].to_numpy(float)
        r_obs, p_perm = _perm_p(x, y)
        _, p_param = pearsonr(x, y)
        rows[ds] = dict(n=int(len(d)), pearson_r=r_obs,
                        parametric_p=float(p_param), permutation_p=p_perm,
                        sign_as_expected=bool(r_obs < 0))

    fdr = st.fdr_dict({ds: v["permutation_p"] for ds, v in rows.items()})
    for ds in rows:
        rows[ds]["permutation_q_fdr"] = fdr[ds]["q"]
        rows[ds]["sig_fdr"] = fdr[ds]["significant_fdr"]

    n_sig = sum(1 for v in rows.values() if v["sig_fdr"] and v["sign_as_expected"])
    n_wrong = sum(1 for v in rows.values() if v["sig_fdr"] and not v["sign_as_expected"])
    out = dict(predictor=PREDICTOR, n_permutations=B, n_datasets=len(rows),
               n_sig_fdr_correct_sign=n_sig, n_sig_fdr_wrong_sign=n_wrong,
               per_dataset=rows,
               note="permutation_p is the exact two-sided null P(|r_perm| >= |r_obs|); "
                    "compare against parametric_p. FDR across datasets on permutation_p.")

    outdir = config.RESULTS_DIR / "experiments"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "exp_E_permutation.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== E: permutation test ({PREDICTOR}, B={B}) ===")
    print(f"  {'dataset':22} {'n':>3} {'r':>7} {'param_p':>10} {'perm_p':>10} {'q_fdr':>8} {'sig':>4}")
    for ds, v in sorted(rows.items(), key=lambda kv: kv[1]["pearson_r"]):
        print(f"  {ds:22} {v['n']:>3} {v['pearson_r']:+7.3f} {v['parametric_p']:>10.4g} "
              f"{v['permutation_p']:>10.4g} {v['permutation_q_fdr']:>8.3f} "
              f"{'Y' if v['sig_fdr'] else '.':>4}")
    print(f"\n  significant after FDR: {n_sig} correct-sign, {n_wrong} wrong-sign "
          f"(of {len(rows)} datasets)")
    return out


if __name__ == "__main__":
    run()
