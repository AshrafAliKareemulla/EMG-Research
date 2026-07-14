"""E10 — senic anomaly investigation.

senic was the lone dataset where distance-to-pool correlated POSITIVELY with LOSO accuracy
(opposite to every other dataset). senic has uneven session counts per subject + electrode-shift /
position / fatigue conditions. This probe tests the likely confound: do subjects with more data /
more sessions appear both more divergent AND easier? Correlates per-subject session/trial counts with
loso_acc and mmd_to_pool (from the Module-5 parquet + the manifest). Mostly local (no frame needed).

Only meaningful for datasets with a Module-5 difficulty parquet; dataset-agnostic otherwise.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config


def run(dataset="senic", seed=42):
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "senic_probe"
    outdir.mkdir(parents=True, exist_ok=True)
    dpath = config.find("module5", f"{dataset}__difficulty.parquet")
    mpath = config.L1_ROOT / dataset / "manifest.parquet"
    if not dpath.exists() or not mpath.exists():
        result = dict(dataset=dataset, note="missing module5 parquet or manifest")
        (outdir / f"{dataset}__senic_probe.json").write_text(json.dumps(result, indent=2))
        return result
    diff = pd.read_parquet(dpath)
    m = pd.read_parquet(mpath)
    # per-subject data-volume descriptors
    vol = m.groupby("subject").agg(n_trials=("trial_key", "count"),
                                   n_sessions=("session", "nunique")).reset_index()
    if "is_fatigue" in m.columns:
        vol = vol.merge(m.groupby("subject")["is_fatigue"].mean().rename("fatigue_frac").reset_index(),
                        on="subject", how="left")
    d = diff.merge(vol, on="subject", how="left")

    # n_trials was a deterministic function of n_sessions -> the old probe reported the SAME
    # correlation twice (identical to 16 dp) and called it two pieces of evidence. Detect and
    # collapse the collinearity instead of pretending they are independent.
    from scipy.stats import pearsonr, spearmanr
    st, ss = d["n_trials"].to_numpy(float), d["n_sessions"].to_numpy(float)
    ok = np.isfinite(st) & np.isfinite(ss)
    collinear = bool(ok.sum() >= 4 and ss[ok].std() > 0 and st[ok].std() > 0
                     and abs(spearmanr(st[ok], ss[ok])[0]) > 0.999)

    def corr(a, b):
        x, y = d[a].to_numpy(float), d[b].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 4 or x[ok].std() < 1e-12 or y[ok].std() < 1e-12:
            return dict(note="undefined (constant or too few subjects)", n=int(ok.sum()))
        r, p = pearsonr(x[ok], y[ok])
        return dict(pearson_r=float(r), p_value=float(p), n=int(ok.sum()))

    def partial_corr(a, b, ctrl):
        """corr(a, b | ctrl) via residualisation. Tests whether the anomalous positive
        loso~mmd association survives after removing per-subject data volume."""
        x, y, z = (d[a].to_numpy(float), d[b].to_numpy(float), d[ctrl].to_numpy(float))
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if ok.sum() < 5 or z[ok].std() < 1e-12:
            return dict(note="undefined (control constant or too few subjects)", n=int(ok.sum()))
        Z = np.column_stack([np.ones(ok.sum()), z[ok]])
        rx = x[ok] - Z @ np.linalg.lstsq(Z, x[ok], rcond=None)[0]
        ry = y[ok] - Z @ np.linalg.lstsq(Z, y[ok], rcond=None)[0]
        if rx.std() < 1e-12 or ry.std() < 1e-12:
            return dict(note="undefined (zero residual variance)", n=int(ok.sum()))
        r, p = pearsonr(rx, ry)
        return dict(pearson_r=float(r), p_value=float(p), n=int(ok.sum()), controlled_for=ctrl)

    raw = corr("loso_acc", "mmd_to_pool")
    part = partial_corr("loso_acc", "mmd_to_pool", "n_sessions")

    # Verdict logic, stated explicitly rather than left to the reader.
    verdict = "inconclusive"
    if "pearson_r" in raw:
        vs, vm = corr("loso_acc", "n_sessions"), corr("mmd_to_pool", "n_sessions")
        both_sig = all("p_value" in v and v["p_value"] < 0.05 for v in (vs, vm))
        same_sign = ("pearson_r" in vs and "pearson_r" in vm
                     and np.sign(vs["pearson_r"]) == np.sign(vm["pearson_r"]))
        if both_sig and same_sign:
            verdict = "data-volume confound supported"
        elif "pearson_r" in part and abs(part["pearson_r"]) < 0.2:
            verdict = "anomaly explained by n_sessions (partial corr collapses)"
        elif "pearson_r" in part and np.sign(part["pearson_r"]) == np.sign(raw["pearson_r"]) \
                and abs(part["pearson_r"]) > 0.3:
            verdict = ("data-volume confound NOT supported: the positive loso~mmd association "
                       "survives controlling for n_sessions; senic remains an unexplained "
                       "sign reversal and must be reported as an outlier, not as a solved case")

    result = dict(
        dataset=dataset, n_subjects=int(len(d)),
        n_sessions_distribution={int(k): int(v) for k, v in
                                 d["n_sessions"].value_counts().sort_index().items()},
        n_trials_collinear_with_n_sessions=collinear,
        loso_acc_vs_n_sessions=corr("loso_acc", "n_sessions"),
        mmd_to_pool_vs_n_sessions=corr("mmd_to_pool", "n_sessions"),
        loso_acc_vs_mmd=raw,
        loso_acc_vs_mmd_partial_given_n_sessions=part,
        verdict=verdict,
        note=("Confound requires loso_acc AND mmd to move the SAME way with session count, "
              "both significantly. If instead the partial correlation of loso~mmd given "
              "n_sessions retains the anomalous sign and magnitude, the confound hypothesis "
              "is refuted and senic stays an outlier."),
    )
    (outdir / f"{dataset}__senic_probe.json").write_text(json.dumps(result, indent=2))
    return result
