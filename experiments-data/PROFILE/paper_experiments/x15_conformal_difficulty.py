"""X15 — Conformal difficulty prediction (NOVEL). A calibrated, coverage-guaranteed difficulty score.

The paper predicts a subject's LOSO accuracy from a cheap training-free statistic (MMD-to-pool). X15
turns that point predictor into a DISTRIBUTION-FREE PREDICTION INTERVAL via split/CV conformal
prediction, and — the honest part — validates the finite-sample marginal coverage guarantee
LEAVE-ONE-COHORT-OUT, i.e. for subjects from a lab never seen in training. So the deployable claim
becomes: "for a new user from an unseen cohort, their true cross-subject accuracy lands in our interval
at least (1 - alpha) of the time, before any model is trained."

NOVELTY: conformal calibration of a training-free difficulty predictor for sEMG. The field reports
point correlations; nobody gives a coverage-guaranteed interval on 'how hard will this user be'.

STATISTICS: split-conformal + cross-conformal prediction; exchangeability; the finite-sample
(ceil((n+1)(1-alpha)))-th order-statistic quantile; empirical coverage vs nominal; interval efficiency
(width); and a contrast with a naive Gaussian-residual interval, which is MIS-CALIBRATED off Gaussian.

GROUND TRUTH: on exchangeable synthetic data with NON-Gaussian (uniform) residuals, the split-conformal
(1-alpha) interval covers ~(1-alpha) exactly, while the naive Gaussian interval is mis-calibrated;
and the finite-sample quantile index is exactly ceil((n+1)(1-alpha)).
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import common

PREDICTORS = ["mmd_to_pool", "hdiv_to_pool", "kl_mean_to_pool", "kl_cov_to_pool"]
Z90 = 1.6448536269514722   # two-sided 90% Gaussian


def _ridge(x, y, alpha=1.0):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha).fit(np.atleast_2d(x), y)
    return lambda xt: m.predict(np.atleast_2d(xt))


def conformal_quantile(residuals, alpha):
    """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest |residual|.

    Guarantees marginal coverage >= 1-alpha under exchangeability. Returns +inf when the calibration
    set is too small to certify the level (an honest, infinitely-wide interval)."""
    r = np.sort(np.asarray(residuals, float))
    n = len(r)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(r[k - 1])


def split_conformal(x_tr, y_tr, x_cal, y_cal, x_te, alpha=0.1, fit=_ridge):
    """Split-conformal intervals: fit on train, calibrate |residual| quantile on cal, band x_te."""
    f = fit(x_tr, y_tr)
    q = conformal_quantile(np.abs(np.asarray(y_cal) - f(x_cal)), alpha)
    pred = f(x_te)
    return pred - q, pred + q, q


def loco_conformal(A, predictors, alpha=0.1, cal_frac=0.4, seed=0):
    """Leave-one-COHORT-out split-conformal. Returns per-cohort empirical coverage + interval width for
    the conformal band and, for contrast, a naive Gaussian band. Target/predictors are z-scored WITHIN
    each dataset so cross-dataset accuracy-scale differences don't break exchangeability of the target.
    """
    rng = np.random.default_rng(seed)
    cohorts = sorted(A.cohort.unique())
    rows = []
    for held in cohorts:
        tr = A[A.cohort != held]
        te = A[A.cohort == held]
        if len(te) < 4 or len(tr) < 20:
            continue
        idx = rng.permutation(len(tr))
        n_cal = max(10, int(cal_frac * len(tr)))
        cal, ptr = tr.iloc[idx[:n_cal]], tr.iloc[idx[n_cal:]]
        Xtr, ytr = ptr[predictors].to_numpy(), ptr["z_acc"].to_numpy()
        Xcal, ycal = cal[predictors].to_numpy(), cal["z_acc"].to_numpy()
        Xte, yte = te[predictors].to_numpy(), te["z_acc"].to_numpy()
        lo, hi, q = split_conformal(Xtr, ytr, Xcal, ycal, Xte, alpha)
        cov_c = float(((yte >= lo) & (yte <= hi)).mean())
        # naive Gaussian band from the calibration residual std
        f = _ridge(Xtr, ytr)
        sig = float(np.std(ycal - f(Xcal)) + common.EPS)
        p = f(Xte)
        cov_g = float(((yte >= p - Z90 * sig) & (yte <= p + Z90 * sig)).mean())
        rows.append(dict(cohort=held, n=len(te), conformal_coverage=cov_c,
                         conformal_width=float(2 * q) if np.isfinite(q) else float("inf"),
                         gaussian_coverage=cov_g, gaussian_width=float(2 * Z90 * sig)))
    if not rows:
        return dict(note="too few cohorts")
    cc = float(np.mean([r["conformal_coverage"] for r in rows]))
    gg = float(np.mean([r["gaussian_coverage"] for r in rows]))
    w = float(np.mean([r["conformal_width"] for r in rows if np.isfinite(r["conformal_width"])]))
    return dict(alpha=alpha, nominal_coverage=1 - alpha, n_cohorts=len(rows),
                mean_conformal_coverage=cc, mean_conformal_width=w,
                mean_gaussian_coverage=gg,
                conformal_valid=bool(cc >= 1 - alpha - 0.05),
                conformal_better_calibrated=bool(abs(cc - (1 - alpha)) <= abs(gg - (1 - alpha))),
                per_cohort=rows)


def _load():
    from dsprofile import config
    rows = []
    for f in config.find_all("module5", "*__difficulty.parquet"):
        ds = Path(f).name.split("__")[0]
        d = pd.read_parquet(f)
        if len(d) < 4 or d["loso_acc"].std() < 1e-9:
            continue
        d = d.copy()
        d["dataset"] = ds
        d["cohort"] = config.COHORTS.get(ds, ds)
        d["z_acc"] = (d["loso_acc"] - d["loso_acc"].mean()) / (d["loso_acc"].std() + 1e-9)
        for p in PREDICTORS:
            if p in d:
                d[p] = (d[p] - d[p].mean()) / (d[p].std() + 1e-9)
        rows.append(d)
    if not rows:
        raise FileNotFoundError("no module5 difficulty parquets; run module 5 first")
    return pd.concat(rows, ignore_index=True)


def run(alpha=0.1, seed=42):
    A = _load()
    have = [p for p in PREDICTORS if p in A.columns]
    out = dict(n_subjects=int(len(A)), n_cohorts=int(A.cohort.nunique()),
               mmd_only=loco_conformal(A, ["mmd_to_pool"], alpha, seed=seed),
               all_predictors=loco_conformal(A, have, alpha, seed=seed),
               method=("leave-one-cohort-out split-conformal; z_acc target; coverage guarantee is "
                       "finite-sample and distribution-free under exchangeability."))
    common.atomic_write_json(common.results_dir("x15") / "conformal.json", out)
    return out


# ------------------------------------------------ ground truth
def selftest(check):
    rng = np.random.default_rng(0)
    alpha = 0.1

    def gen(n):
        x = rng.uniform(-2, 2, (n, 1))
        y = 1.5 * x[:, 0] + rng.uniform(-1.5, 1.5, n)      # NON-Gaussian (uniform) residuals
        return x, y

    cov_c, cov_g = [], []
    for _ in range(40):
        xtr, ytr = gen(250)
        xcal, ycal = gen(250)
        xte, yte = gen(600)
        lo, hi, q = split_conformal(xtr, ytr, xcal, ycal, xte, alpha)
        cov_c.append(((yte >= lo) & (yte <= hi)).mean())
        f = _ridge(xtr, ytr)
        sig = np.std(ycal - f(xcal))
        p = f(xte)
        cov_g.append(((yte >= p - Z90 * sig) & (yte <= p + Z90 * sig)).mean())
    mc, mg = float(np.mean(cov_c)), float(np.mean(cov_g))
    check("X15 split-conformal 90% interval covers ~90% (exact calibration)", abs(mc - 0.9) < 0.03, f"{mc:.3f}")
    check("X15 conformal coverage >= nominal (valid guarantee)", mc >= 1 - alpha - 0.02, f"{mc:.3f}")
    check("X15 naive Gaussian band is MIS-calibrated off Gaussian (why conformal is needed)",
          abs(mg - 0.9) > 0.03 and abs(mc - 0.9) < abs(mg - 0.9), f"gauss={mg:.3f} conf={mc:.3f}")
    # finite-sample quantile index: n=100, alpha=0.1 -> k=ceil(101*0.9)=91 -> 91st order statistic
    q100 = conformal_quantile(np.arange(1, 101.0), 0.1)
    check("X15 finite-sample conformal quantile index correct", q100 == 91.0, f"q={q100}")
    # too-small calibration set -> honest infinite interval
    check("X15 undersized calibration -> +inf (honest, cannot certify)",
          not np.isfinite(conformal_quantile(np.arange(1, 6.0), 0.01)))
