"""X11 — Meta-regression of the difficulty effect. Complements X1 at the between-dataset level (§6).

Models each dataset's Fisher-z(difficulty r) on dataset properties (mean accuracy, class count, channel
count), weighted by n and with COHORT-clustered bootstrap CIs, to formally test how much of the
difficulty correlation is accuracy-moderated (the floor effect) across corpora.

Reads the committed per-dataset results (no feature frames). GROUND TRUTH: simulate per-dataset r's
from r_k = tanh(beta*mean_acc_k + noise) with known beta -> the regression recovers beta's sign and a
CI covering it; a null (beta=0) -> CI covers 0.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from . import common


def _wls_coef(X, z, w, col):
    """Weighted least squares; return the coefficient of predictor column `col` (design has intercept)."""
    X = np.asarray(X, float)
    z = np.asarray(z, float)
    sw = np.sqrt(np.asarray(w, float))
    beta, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)
    return float(beta[col])


def collect():
    """One row per dataset: difficulty r + n + mean_acc + n_classes + n_channels + cohort."""
    from dsprofile import config
    rows = []
    for f in config.find_all("module5", "*__difficulty.json"):
        d = json.loads(Path(f).read_text())
        ds = d.get("dataset") or Path(f).name.split("__")[0]
        pc = d.get("predictor_correlations", {}).get(config.PRIMARY_PREDICTOR, {})
        if "pearson_r" not in pc or d.get("n_subjects", 0) < 5:
            continue
        m2 = config.find("module2", f"{ds}__separability.json")
        m2d = json.loads(m2.read_text()) if m2.exists() else {}
        rows.append(dict(dataset=ds, cohort=config.COHORTS.get(ds, ds),
                         r=float(pc["pearson_r"]), n=int(d["n_subjects"]),
                         mean_acc=float(d.get("loso_acc_mean", np.nan)),
                         n_classes=m2d.get("n_classes"), n_channels=m2d.get("n_channels")))
    return [r for r in rows if np.isfinite(r["r"]) and np.isfinite(r["mean_acc"])]


def meta_regress(rows, predictor="mean_acc", B=5000, seed=0):
    if len(rows) < 5:
        return dict(note="need >=5 datasets", k=len(rows))
    keep = [r for r in rows if r.get(predictor) is not None and np.isfinite(r.get(predictor, np.nan))]
    if len(keep) < 5:
        return dict(note=f"too few datasets with {predictor}", k=len(keep))
    z = np.arctanh(np.clip([r["r"] for r in keep], -0.999999, 0.999999))
    x = np.array([r[predictor] for r in keep], float)
    xz = (x - x.mean()) / (x.std() + common.EPS)          # standardise the predictor
    w = np.array([r["n"] - 3 for r in keep], float)
    X = np.column_stack([np.ones(len(keep)), xz])

    def stat(rws):
        idx = [keep.index(r) for r in rws]
        return _wls_coef(X[idx], z[idx], w[idx], col=1)

    boot = common.cluster_bootstrap([dict(cohort=r["cohort"], _row=r) for r in keep],
                                    lambda rr: _coef_from(rr, X, z, w, keep), "cohort", B=B, seed=seed)
    coef = stat(keep)

    # SIGN CONVENTION (the old hardcoded string had this BACKWARDS and was pasted onto every
    # predictor). The outcome is Fisher-z(r) and the difficulty r is NEGATIVE. So a coefficient
    # that is POSITIVE pushes z(r) toward 0 => the negative correlation gets WEAKER; a NEGATIVE
    # coefficient makes it MORE negative => STRONGER. Generated from the real number, per predictor.
    direction = "WEAKER (less negative)" if coef > 0 else "STRONGER (more negative)"
    sig = "CI excludes 0" if boot["excludes_zero"] else "CI includes 0 -> no evidence of moderation"
    interp = (f"coef_fisher_z = {coef:+.3f} on standardised {predictor}: as {predictor} increases, "
              f"the per-dataset difficulty correlation becomes {direction}. {sig}. "
              + ("A POSITIVE coef here IS the between-dataset floor effect (higher-accuracy datasets "
                 "have a weaker predictor)." if predictor == "mean_acc" else ""))
    return dict(k=int(len(keep)), predictor=predictor, coef_fisher_z=coef,
                cohort_clustered_ci95=boot["ci95"], excludes_zero=boot["excludes_zero"],
                n_cohorts=boot.get("n_clusters"),
                outcome="fisher_z(difficulty_r)  [r is NEGATIVE; coef>0 => weaker predictor]",
                interpretation=interp)


def _coef_from(rws, X, z, w, keep):
    idx = [keep.index(r["_row"]) for r in rws]
    return _wls_coef(X[idx], z[idx], w[idx], col=1)


def run(seed=42):
    rows = collect()
    out = dict(n_datasets=len(rows), rows=rows,
               vs_mean_acc=meta_regress(rows, "mean_acc", seed=seed),
               vs_n_classes=meta_regress(rows, "n_classes", seed=seed),
               vs_n_channels=meta_regress(rows, "n_channels", seed=seed))
    common.atomic_write_json(common.results_dir("x11") / "meta_regression.json", out)
    return out


# ------------------------------------------------ ground truth
def _sim_rows(beta, k=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(k):
        ma = rng.uniform(0.1, 0.7)
        zt = beta * ((ma - 0.4) / 0.2) + 0.25 * rng.standard_normal()
        rows.append(dict(dataset=f"d{i}", cohort=f"c{i}", r=float(np.tanh(zt)),
                         n=int(rng.integers(20, 45)), mean_acc=ma,
                         n_classes=int(rng.integers(5, 50)), n_channels=int(rng.integers(4, 16))))
    return rows


def selftest(check):
    neg = meta_regress(_sim_rows(-0.8, seed=1), "mean_acc", B=1500, seed=1)
    check("X11 recovers a NEGATIVE accuracy-moderation coefficient", neg["coef_fisher_z"] < 0,
          f"coef={neg['coef_fisher_z']:.3f} ci={neg['cohort_clustered_ci95']}")
    # aggregate a few null sims: CI should cover 0 the large majority of the time
    covers = 0
    for s in range(8):
        nul = meta_regress(_sim_rows(0.0, seed=100 + s), "mean_acc", B=800, seed=s)
        lo, hi = nul["cohort_clustered_ci95"]
        covers += int(lo <= 0 <= hi)
    check("X11 null: cohort-clustered CI covers 0 in >=6/8 sims", covers >= 6, f"{covers}/8")
